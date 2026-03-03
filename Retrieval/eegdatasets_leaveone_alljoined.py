"""
PyTorch Dataset for Alljoined1 with leave-one-subject-out split support.

Dataset: Alljoined1
Experiment: Generalisability

Loads preprocessed EEG recordings, ViT-H-14 CLIP image embeddings, and COCO scene
category labels for the Alljoined1 naturalistic scene dataset (NSD images). Uses the
NSD experiment design matrix to map trial indices to stimulus images. Image paths and
dataset configuration are resolved via configs/alljoined.json.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import pandas as pd

cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'
import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)

import json
import scipy

# Load dataset paths from the project-level config (Alljoined1 = generalisability dataset)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(_PROJECT_ROOT, "configs", "alljoined.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = os.path.normpath(os.path.join(_PROJECT_ROOT, config["data_path"]))
img_directory_training = os.path.normpath(os.path.join(_PROJECT_ROOT, config["img_directory_training"]))
img_directory_test = os.path.normpath(os.path.join(_PROJECT_ROOT, config["img_directory_test"]))
expdesign = scipy.io.loadmat(os.path.normpath(os.path.join(_PROJECT_ROOT, config["nsd_expdesign"])))
label_data = os.path.normpath(os.path.join(_PROJECT_ROOT, config["label_json_path"]))

class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, debug = True, exclude_subject=None, subjects=None, train=True, time_window=[0, 1], classes=None, pictures=None, val_size=None):
        self.data_path = data_path
        self.debug = debug
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 80 
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
       
        assert any(sub in self.subject_list for sub in self.subjects)

        _alljoined_img_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, "Data", "images_set", "Alljoined1"))
        with open(os.path.join(_alljoined_img_dir, "cocoid_to_categories.json"), "r") as f:
            self.cocoid_to_categories = json.load(f)
        with open(os.path.join(_alljoined_img_dir, "categories.json"), "r") as l:
            categories = json.load(l)
        # Create a sorted list of category names for consistent, reproducible ordering
        self.category_list = sorted(categories["categories"])
        # Build a lookup dict mapping each category name to a unique integer index
        # This will let us convert category strings into numeric labels easily
        self.category_to_index = {cat: idx for idx, cat in enumerate(self.category_list)}

        # Load EEG, labels
        self.data, self.labels, self.img , self.text = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)

        
        # Feature extraction and filtering

        if self.classes is None and self.pictures is None:
            features_filename = f'ViT-H-14_features_alljoined_train.pt' if self.train else f'ViT-H-14_features_alljoined_test.pt'
            use_cache = os.path.exists(features_filename)

            if use_cache:
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']

            if not use_cache:
                    print(f"[EEGDataset] Generating fresh features → {features_filename}")
                    if self.debug == True:
                        print(f"[DEBUG] Final self.text length before encoding: {len(self.text)}")
                        print(f"[DEBUG] First 5 text prompts: {self.text[:5]}")
                    self.text_features = self.Textencoder(self.text)
                    self.img_features = self.ImageEncoder(self.img)
                    torch.save({
                        'text_features': self.text_features.cpu(),
                        'img_features': self.img_features.cpu(),
                    }, features_filename)
        else:
            print("[EEGDataset] Feature caching disabled; generating from scratch.")
            if self.debug == True:
                print(f"[DEBUG] Final self.text length before encoding: {len(self.text)}")
                print(f"[DEBUG] First 5 text prompts: {self.text[:5]}")
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)
        

        if self.debug == True:
            print(f"[DEBUG] Final dataset sizes — EEG: {len(self.data)}, IMG: {len(self.img)}, LABELS: {len(self.labels)}")

                
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
            # Iterate through each session folder in the subject’s image directory (sorted for consistency)
            for ses_folder in sorted(os.listdir(subject_img_dir)):
                ses_path = os.path.join(subject_img_dir, ses_folder)
                if os.path.isdir(ses_path):
                    try:
                        session_id = int(ses_folder.split("-")[-1])  # e.g., ses-01 → 1
                    except Exception as e:
                        continue
                    trial_to_path = {
                        # Build a mapping from trial number → image file path for this session
                        int(f.split("_")[0]): os.path.join(ses_path, f)
                        for f in sorted(os.listdir(ses_path))
                        if f.lower().endswith(('.jpg', '.jpeg')) and os.path.isfile(os.path.join(ses_path, f))
                    }
                    # Store the trial→path mapping under this session ID
                    session_image_lookup[session_id] = trial_to_path

            # === Match EEG rows to image paths and labels ===
            for _, row in df.iterrows():
                # Extract trial and session identifiers from the EEG dataframe row
                trial = int(row["trial_fixed"])
                session = int(row["session"])
                # Convert the EEG signal (list/array) into a FloatTensor of shape [channels, timepoints]
                eeg = torch.tensor(np.stack(row["EEG"]).astype(np.float32))

                 # Try to obtain the image path from the matching session.
                session_dict = session_image_lookup.get(session, {})
                img_path = session_dict.get(trial)
                 # If there's only one session folder, fall back to that mapping
                if not img_path and len(session_image_lookup) == 1:
                    only_session = list(session_image_lookup.keys())[0]
                    img_path = session_image_lookup[only_session].get(trial)
                 # Skip this trial if no matching image was found 
                if not img_path:
                        print(f"[WARN] No matching image for subject {subject}, session {session}, trial {trial}")

                # Append EEG tensor and corresponding image file path to our lists
                data_list.append(eeg)
                images.append(img_path)

                # Assign label based on coco_id → category
                # The COCO images can belong to multiple categories; you want to decide on one label per trial. 
                # By choosing the least-used category, you avoid over-representing some classes in your EEG–image pairing.
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
        n_timepoints = 334
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

        return data_tensor, label_tensor, images, texts


    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        indices = (self.times >= start) & (self.times <= end)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]

        return extracted_data
    
    def Textencoder(self, text):   
            print(f"[DEBUG] Number of text prompts received: {len(text)}")
            for i, t in enumerate(text):
                print(f"{i+1:02d}: {t}")
            text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)
            
            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)
            
            text_features = F.normalize(text_features, dim=-1).detach()
       
            return text_features
        
    def ImageEncoder(self,images):
        batch_size = 20  
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        
        return image_features
    #
    def __getitem__(self, index):
        # Get EEG data and corresponding class label
        x = self.data[index]
        label = self.labels[index]

        # Convert label to int (in case it's a tensor)
        label_idx = label.item() if isinstance(label, torch.Tensor) else label

        # Safely map to text and image features based on class label
        text = self.text[label_idx]  # e.g., "This picture contains a dog"
        text_feat = self.text_features[label_idx]
        img_feat = self.img_features[index]

        # Optionally, select an example image for this class
        # This assumes self.img holds one or more images per class
        img = self.img[index]
        return x, label, text, text_feat, img, img_feat

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
    
    
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
