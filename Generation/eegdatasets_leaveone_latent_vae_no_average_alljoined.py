"""
PyTorch Dataset for Alljoined1: loads EEG, SDXL VAE latents, and COCO scene categories.

Dataset: Alljoined1
Experiment: Generalisability

Implements a leave-one-subject-out Dataset for naturalistic scene EEG from Alljoined1
(NSD stimuli), pairing per-trial EEG with precomputed SDXL VAE latents and COCO
super-category labels for low-level reconstruction training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import pandas as pd 
from diffusers import AutoencoderKL


import json

#Reduce memory usage
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load CLIP model for text feature extraction

# Load VAE for latent extraction (outputs latents of shape (batch, 4, 64, 64)) 
# VAE for extracting latent representations from stimulus images.
# Download the SD2.1 VAE weights and place them at Generation/stable-diffusion-2-1-vae/
# (see README for instructions — model not included in repo due to size)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_vae_dir = os.path.join(_PROJECT_ROOT, "Generation", "stable-diffusion-2-1-vae")
vae = AutoencoderKL.from_pretrained(local_vae_dir, subfolder="None").to(device)
vae.eval()

# Preprocessing pipeline for images into VAE input
deeva_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset paths from the project-level config (Alljoined1 = generalisability dataset)
config_path = os.path.join(_PROJECT_ROOT, "configs", "alljoined.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = os.path.normpath(os.path.join(_PROJECT_ROOT, config["data_path"]))
img_directory_training = os.path.normpath(os.path.join(_PROJECT_ROOT, config["img_directory_training"]))
img_directory_test = os.path.normpath(os.path.join(_PROJECT_ROOT, config["img_directory_test"]))

features_path = os.path.normpath(os.path.join(_PROJECT_ROOT, config["features_path"]))

class EEGDataset():
    
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']

    def __init__(self, data_path, exclude_subject=None, subjects=None, debug = False, train=True, time_window=[0, 1.0], classes = None, pictures = None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.debug = debug
        self.n_cls = 80
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  # 保存这个参数

        print("Subjects expected:", self.subjects)
        print("Subjects available:", self.subject_list)
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        _alljoined_img_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, "Data", "images_set", "Alljoined1"))
        with open(os.path.join(_alljoined_img_dir, "cocoid_to_categories.json"), "r") as f:
            self.cocoid_to_categories = json.load(f)
        with open(os.path.join(_alljoined_img_dir, "categories.json"), "r") as l:
            categories = json.load(l)

        self.category_list = sorted(categories["categories"])
        self.category_to_index = {cat: idx for idx, cat in enumerate(self.category_list)}
        
        self.data, self.labels, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)

        if self.classes is None and self.pictures is None:
            # Try to load the saved features if they exist
            features_folder = features_path
            features_filename = os.path.join(features_folder, 'train_image_latent_512_alljoined.pt') if self.train else os.path.join(features_folder, 'test_image_latent_512_alljoined.pt')

            if os.path.exists(features_filename) :
                saved_features = torch.load(features_filename)
                self.img_features = saved_features.get('img_features', None)
            else:
                print(f"Error: Features not found and {features_filename} ! ")
                print("Generating features from scratch:")
                self.img_features = self.ImageEncoder(self.img)
                torch.save({ "img_features": self.img_features }, features_filename) 
        else:
            self.img_features = self.ImageEncoder(self.img)
            
    def load_data(self):
        """
        Loads EEG and image data for selected subjects.
        Returns:
        data_tensor: Stacked EEG data (as a torch.Tensor)
        label_tensor: Stacked label indices (as a torch.Tensor)
        texts: A list of text prompts (one per category)
        images: A list of image file paths
        """
        import random

        data_list, label_list, images, texts = [], [], [], []
        
        # Choose the root for images based on training or testing mode.
        img_root = img_directory_training if self.train else img_directory_test
        
        if self.debug:
            print("[DEBUG] Subjects:", self.subjects)
            print("[DEBUG] Exclude subject:", self.exclude_subject)

        # Create text prompts from the sorted category list (one for each category)
        texts = [f"This picture contains a {cat}" for cat in self.category_list]

        # Initialize a counter for each category to track assignment frequency
        category_counter = {cat: 0 for cat in self.category_list}
        
        # Iterate over subjects
        for subject in self.subjects:
            if self.train and subject == self.exclude_subject:
                continue

             # Path to the EEG file (parquet format)
            subject_dir = os.path.join(self.data_path, subject)
            eeg_filename = f"{'train' if self.train else 'test'}_{subject}.parquet"
            eeg_path = os.path.join(subject_dir, eeg_filename)

            if not os.path.exists(eeg_path):
                print(f"[WARN] Missing EEG file for {subject}: {eeg_path}")
                continue

            df = pd.read_parquet(eeg_path)

            # Build a session_image_lookup: maps each session number to a trial -> image path dictionary.
            subject_img_dir = os.path.join(img_root, subject)
            session_image_lookup = {}

            for ses_folder in sorted(os.listdir(subject_img_dir)):
                ses_path = os.path.join(subject_img_dir, ses_folder)
                if os.path.isdir(ses_path):
                    try:
                        session_id = int(ses_folder.split("-")[-1])  # e.g., ses-01 → 1
                    except Exception as e:
                        continue
                    trial_to_path = {
                        int(f.split("_")[0]): os.path.join(ses_path, f)
                        for f in sorted(os.listdir(ses_path))
                        if f.lower().endswith(('.jpg', '.jpeg')) and os.path.isfile(os.path.join(ses_path, f))
                    }
                    session_image_lookup[session_id] = trial_to_path

            # === Match EEG rows to image paths and labels ===
            for _, row in df.iterrows():
                trial = int(row["trial_fixed"])
                session = int(row["session"])
                eeg = torch.tensor(np.stack(row["EEG"]).astype(np.float32))

                 # Try to obtain the image path from the matching session.
                session_dict = session_image_lookup.get(session, {})
                img_path = session_image_lookup.get(session, {}).get(trial)
                if not img_path and len(session_image_lookup) == 1:
                    only_session = list(session_image_lookup.keys())[0]
                    img_path = session_image_lookup[only_session].get(trial)
                if not img_path:
                    if self.debug:
                        print(f"[WARN] No matching image for subject {subject}, session {session}, trial {trial}")
                    continue

                data_list.append(eeg)
                images.append(img_path)

                # Assign label based on coco_id → category
                coco_id = int(row["coco_id"])
                candidate_categories = self.cocoid_to_categories.get(str(coco_id), {}).get("categories", [])
                if candidate_categories:
                    # Select the candidate with the lowest count so far
                    chosen_category = min(candidate_categories, key=lambda cat: category_counter.get(cat, 0))
                    category_counter[chosen_category] += 1
                    label = self.category_to_index.get(chosen_category, -1)
                else:
                    # Fallback: use the first category in your sorted list
                    chosen_category = self.category_list[0]
                    category_counter[chosen_category] += 1
                    label = self.category_to_index.get(chosen_category, -1)
                label_list.append(torch.tensor([label], dtype=torch.long))

        # Set shared EEG time axis and channel names
        n_timepoints = 334 #334
        sampling_rate = 512
        self.times = torch.tensor(np.arange(n_timepoints) / sampling_rate, dtype=torch.float32)
        self.ch_names = [ 
            'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 
            'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 
            'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 
            'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 
            'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 
            'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
        ]

        data_tensor = torch.stack(data_list)
        label_tensor = torch.stack(label_list)

        return data_tensor, label_tensor, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        indices = (self.times >= start) & (self.times <= end)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]

        return extracted_data
    """
    def Textencoder(self, text):   
      
        print(f"[DEBUG] Number of text prompts received: {len(text)}")
        for i, t in enumerate(text):
            print(f"{i+1:02d}: {t}")
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)
        
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
       
        return text_features
    """
        
    def ImageEncoder(self, images):
        """
        Given a list of image file paths, returns a Tensor of shape (len(images), 4, 64, 64)
        containing VAE latents.
        """
        latents = []
        for path in images:
            img = Image.open(path).convert("RGB")
            pix = deeva_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                enc = vae.encode(pix)
                z = enc.latent_dist.sample()
            latents.append(z.cpu().squeeze(0))
        return torch.stack(latents, dim=0)      # (N,4,64,64)

    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]

        # Convert label to int (in case it's a tensor)
        label_idx = label.item() if isinstance(label, torch.Tensor) else label

        # Safely map to text and image features based on class label
      
        img_feat = self.img_features[label_idx]

        img = self.img[label_idx % len(self.img)] if len(self.img) > label_idx else None

        return x, label, img, img_feat
    
    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    # data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"  # Replace with the path to your data
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True)    
    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
    # train_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=True)    
    # test_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=False)    
    # train_dataset = EEGDataset(data_path, train=True) 
    # test_dataset = EEGDataset(data_path, train=False) 
    # 训练的eeg数据：torch.Size([16540, 4, 17, 100]) [训练图像数量，训练图像重复数量，通道数，脑电信号时间点]
    # 测试的eeg数据：torch.Size([200, 80, 17, 100])
    # 1秒 'times': array([-0.2 , -0.19, -0.18, ... , 0.76,  0.77,  0.78, 0.79])}
    # 17个通道'ch_names': ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            

