import torch
import pprint

file_path = r"D:\Universität\Master\4. Semester\Forschungspraxis\fp_python\Data\Raw\EEG ImageNet\EEG-ImageNet_1.pth"
checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
pprint.pprint(checkpoint)