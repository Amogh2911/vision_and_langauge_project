import os
import h5py
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from torch import nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, patch_size=4):
        super(FeatureExtractor, self).__init__()
        # Visual feature extraction using ResNet18
        self.img_feature_extractor = models.resnet18(pretrained=True)
        self.img_feature_extractor = nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))
        self.resize_dim = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        
    def forward(self, img):
        outputs = self.resize_dim(self.img_feature_extractor(img))
        return outputs

# Dataset paths
folder_head = '/content/drive/MyDrive/vision_and_language_project/Surgical-VQLA/dataset/EndoVis-17-VQLA/left_frames/'
folder_tail = '*.jpg'

# Collect all image files
filenames = glob(folder_head + folder_tail)
print(f"Found {len(filenames)} images.")

# Filter filenames to include all images (or modify filter logic as needed)
new_filenames = []
for filename in filenames:
    frame_num = int(filename.split('/')[-1].split('_')[-1].split('.')[0].strip('frame'))
    if frame_num % 1 == 0:  # Keep all frames
        new_filenames.append(filename)

print(f"Processing {len(new_filenames)} images...")

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((300, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set the patch size directly
patch_size = 5

# Declare feature extraction model
feature_network = FeatureExtractor(patch_size=patch_size)

# Set data parallel if GPU is available
if torch.cuda.is_available():
    print("Using GPU for feature extraction.")
    feature_network = nn.DataParallel(feature_network).cuda()
else:
    print("No GPU found. Using CPU.")
    feature_network = feature_network

feature_network.eval()

# Loop over all images
for img_loc in tqdm(new_filenames):
    # Get visual features
    img = Image.open(img_loc)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    
    with torch.no_grad():
        visual_features = feature_network(img)
        visual_features = torch.flatten(visual_features, start_dim=2)
        visual_features = visual_features.permute((0, 2, 1))   
        visual_features = visual_features.squeeze(0).cpu().numpy()

    # Save extracted features
    img_loc_split = img_loc.split('/')
    save_dir = os.path.join(
        '/content/drive/MyDrive/vision_and_language_project/Surgical-VQLA/dataset/EndoVis-17-VQLA/vqla/img_features/',
        f"{patch_size}x{patch_size}"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{img_loc_split[-1].split('.')[0]}.hdf5")
    with h5py.File(save_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('visual_features', data=visual_features)

    print(f"Saved features to: {save_path}")
