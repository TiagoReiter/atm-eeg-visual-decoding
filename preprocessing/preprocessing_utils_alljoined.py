import torch
import torch.linalg as LA
from tqdm import tqdm

def epoching(args, data_part, seed):
	"""This function first converts the EEG data to MNE raw format, and
	performs channel selection, epoching, baseline correction and frequency
	downsampling. Then, it sorts the EEG data of each session according to the
	image conditions.

	Parameters
	----------
	args : Namespace
		Input arguments.
	data_part : str
		'test' or 'training' data partitions.
	seed : int
		Random seed.

	Returns
	-------
	epoched_data : list of float
		Epoched EEG data.
	img_conditions : list of int
		Unique image conditions of the epoched and sorted EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import mne
	import numpy as np
	from sklearn.utils import shuffle

	chan_order = [ 'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 
    				'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 
					'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 
					'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 
					'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 
					'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'] #Channel names orderd according to the standard 10-30 order, seen in the preprocessing notebook of alljoined1

	### Loop across data collection sessions ###
	epoched_data = []
	img_conditions = []
	for s in range(args.n_ses):

		### Load the EEG data and convert it to MNE raw format ###
		eeg_dir = os.path.join(f"sub-{format(args.sub, '02d')}", f"ses-{format(s+1, '02d')}", f"raw_eeg_{data_part}.npy")
		eeg_data = np.load(os.path.join(args.project_dir, eeg_dir), allow_pickle=True).item () 
		ch_names = eeg_data['ch_names']
		sfreq = eeg_data['sfreq']
		ch_types = eeg_data['ch_types']
		eeg_data = eeg_data['raw_eeg_data']
		# Convert to MNE raw format
		info = mne.create_info(ch_names, sfreq, ch_types)
		raw = mne.io.RawArray(eeg_data, info)
		del eeg_data

		### Get events, drop unused channels and reject target trials ###
		events = mne.find_events(raw, stim_channel='stim')
		# # Select only occipital (O) and posterior (P) channels
		# chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
		# 	'^O *|^P *'))
		# new_chans = [raw.info['ch_names'][c] for c in chan_idx]
		# raw.pick_channels(new_chans)
		# * chose all channels
		raw.pick_channels(chan_order, ordered=True)
		# Reject the target trials (event 99999)
		idx_target = np.where(events[:,2] == 99999)[0]
		events = np.delete(events, idx_target, 0)
		### Epoching, baseline correction and resampling ###
		# * [0, 1.0]
		epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None,0),
			preload=True)
		# epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
		# 	preload=True)
		del raw
		# Resampling
		if args.sfreq < 1000:
			epochs.resample(args.sfreq)
		ch_names = epochs.info['ch_names']
		times = epochs.times

		### Sort the data ###
		data = epochs.get_data()
		events = epochs.events[:,2]
		img_cond = np.unique(events)
		del epochs
		# Select only a maximum number of EEG repetitions
		if data_part == 'test':
			max_rep = 20
		else:
			max_rep = 2
		# Sorted data matrix of shape:
		# Image conditions × EEG repetitions × EEG channels × EEG time points
		sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
			data.shape[2]))
		for i in range(len(img_cond)):
			# Find the indices of the selected image condition
			idx = np.where(events == img_cond[i])[0]
			# Randomly select only the max number of EEG repetitions
			max_rep = min(max_rep, len(idx)) ## fixed so max_rep is not larger than the number of repetitions
			idx = shuffle(idx, random_state=seed, n_samples=max_rep)
			sorted_data[i] = data[idx]
		del data
		epoched_data.append(sorted_data[:, :, :, 50:])
		img_conditions.append(img_cond)
		del sorted_data

	### Output ###
	return epoched_data, img_conditions, ch_names, times

def mvnn(args, epoched_test, epoched_train):
	"""Compute the covariance matrices of the EEG data (calculated for each
	time-point or epoch/repetitions of each image condition), and then average
	them across image conditions and data partitions. The inverse of the
	resulting averaged covariance matrix is used to whiten the EEG data
	(independently for each session).
	
	zero-score standardization also has well performance

	Parameters
	----------
	args : Namespace
		Input arguments.
	epoched_test : list of floats
		Epoched test EEG data.
	epoched_train : list of floats
		Epoched training EEG data.

	Returns
	-------
	whitened_test : list of float
		Whitened test EEG data.
	whitened_train : list of float
		Whitened training EEG data.

	"""

	from sklearn.discriminant_analysis import _cov
	import scipy

	### Loop across data collection sessions ###
	whitened_test = []
	whitened_train = []
	for s in range(args.n_ses):
		session_data = [epoched_test[s], epoched_train[s]]

		### Compute the covariance matrices ###
		# Data partitions covariance matrix of shape:
		# Data partitions × EEG channels × EEG channels
		sigma_part = np.empty((len(session_data),session_data[0].shape[2],
			session_data[0].shape[2]))
		for p in range(sigma_part.shape[0]):
			# Image conditions covariance matrix of shape:
			# Image conditions × EEG channels × EEG channels
			sigma_cond = np.empty((session_data[p].shape[0],
				session_data[0].shape[2],session_data[0].shape[2]))
			for i in tqdm(range(session_data[p].shape[0])):
				cond_data = session_data[p][i]
				# Compute covariace matrices at each time point, and then
				# average across time points
				if args.mvnn_dim == "time":
					sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
						shrinkage='auto') for t in range(cond_data.shape[2])],
						axis=0)
				# Compute covariace matrices at each epoch (EEG repetition),
				# and then average across epochs/repetitions
				elif args.mvnn_dim == "epochs":
					sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
						shrinkage='auto') for e in range(cond_data.shape[0])],
						axis=0)
			# Average the covariance matrices across image conditions
			sigma_part[p] = sigma_cond.mean(axis=0)
		# # Average the covariance matrices across image partitions
		# sigma_tot = sigma_part.mean(axis=0)
		# ? It seems not fair to use test data for mvnn, so we change to just use training data
		sigma_tot = sigma_part[1]
		# Compute the inverse of the covariance matrix
		sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

		### Whiten the data ###
		whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
			session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
			@ sigma_inv).swapaxes(1, 2), session_data[0].shape))
		whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
			session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
				@ sigma_inv).swapaxes(1, 2), session_data[1].shape))

	### Output ###
	return whitened_test, whitened_train


def save_prepr(args, whitened_test, whitened_train, img_conditions_train,
	ch_names, times, seed):
	"""Merge the EEG data of all sessions together, shuffle the EEG repetitions
	across sessions and reshape the data to the format:
	Image conditions × EEG repetitions × EEG channels × EEG time points.
	Then, the data of both test and training EEG partitions is saved.

	Parameters
	----------
	args : Namespace
		Input arguments.
	whitened_test : list of ndarray
		List of whitened test EEG data arrays from different sessions.
	whitened_train : list of ndarray
		List of whitened training EEG data arrays from different sessions.
	img_conditions_train : list of ndarray
		List of image condition labels for training data across sessions.
	ch_names : list of str
		EEG channel names.
	times : ndarray
		EEG time points.
	seed : int
		Random seed.
	"""

	import pickle

	### ✅ FIXED: Merge test data using `np.concatenate()` ###
	merged_test = np.concatenate(whitened_test, axis=1)  # Merge across sessions along the second axis
	del whitened_test

	# Shuffle test repetitions across sessions
	idx = shuffle(np.arange(merged_test.shape[1]), random_state=seed)
	merged_test = merged_test[:, idx]

	# Store test data in a dictionary
	test_dict = {
		'preprocessed_eeg_data': merged_test,
		'ch_names': ch_names,
		'times': times
	}
	del merged_test

	# Define saving directories
	save_dir = os.path.join(args.project_dir, 'Preprocessed_data_250Hz', f'sub-{args.sub:02}')
	file_name_test = 'preprocessed_eeg_test.npy'
	file_name_train = 'preprocessed_eeg_training.npy'
	
	# Ensure the directory exists
	os.makedirs(save_dir, exist_ok=True)

	# Save test data
	with open(os.path.join(save_dir, file_name_test), 'wb') as f:
		pickle.dump(test_dict, f, protocol=4)
	del test_dict

	### ✅ FIXED: Merge training data using `np.concatenate()` ###
	white_data = np.concatenate(whitened_train, axis=0)  # Merge along the first axis (image conditions)
	img_cond = np.concatenate(img_conditions_train, axis=0)  # Merge condition labels
	del whitened_train, img_conditions_train

	### ✅ FIXED: Properly group EEG data by condition ###
	unique_conditions = np.unique(img_cond)
	merged_train = []

	for cond in unique_conditions:
		# Find the indices of the selected condition
		idx = np.where(img_cond == cond)[0]

		# Select all repetitions for this condition
		ordered_data = white_data[idx]
		merged_train.append(ordered_data)

	# Convert list to NumPy array with `dtype=object` for variable-length handling
	merged_train = np.array(merged_train, dtype=object)

	# Shuffle repetitions within each condition
	for i in range(len(merged_train)):
		merged_train[i] = merged_train[i][shuffle(np.arange(len(merged_train[i])), random_state=seed)]

	# Store training data in a dictionary
	train_dict = {
		'preprocessed_eeg_data': merged_train,
		'ch_names': ch_names,
		'times': times
	}
	del merged_train

	# Save training data
	with open(os.path.join(save_dir, file_name_train), 'wb') as f:
		pickle.dump(train_dict, f, protocol=4)
	del train_dict

