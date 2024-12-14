import os
import h5py
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from torch import nn
from visual_bert.modeling_frcnn import GeneralizedRCNN
from visual_bert.utils import Config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature extraction model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(device)
        
    def forward(self, images_mine):
        output_dict = self.frcnn(
            images_mine,
            torch.tensor([[800, 1206]]),
            scales_yx=torch.tensor([[1.2800, 1.0614]]),
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        outputs = output_dict.get("roi_features")
        return outputs

# Dataset path
folder_head = '/content/drive/MyDrive/vision_and_language_project/Surgical-VQLA/dataset/EndoVis-17-VQLA/left_frames/'
folder_tail = '*.jpg'

# Collect filenames
filenames = glob(folder_head + folder_tail)
print(f"Found {len(filenames)} images.")

# Filter by sequence (if needed)
new_filenames = []
for filename in filenames:
    frame_num = int(filename.split('/')[-1].split('_')[-1].split('.')[0].strip('frame'))
    if frame_num % 1 == 0:  # Keep all frames
        new_filenames.append(filename)

print(f"Filtered images: {len(new_filenames)}")

# Feature extraction model
feature_network = FeatureExtractor()

# Set data parallel if GPU is available
if torch.cuda.device_count() > 0:
    feature_network = nn.DataParallel(feature_network)

feature_network = feature_network.to(device)
feature_network.eval()

# Process each image
for img_loc in tqdm(new_filenames):
    # Load and preprocess the image
    img = Image.open(img_loc)
    raw_sizes = torch.tensor(np.array(img).shape[:2])
    pil_image = img.resize((1206, 800), Image.BILINEAR)
    normalizer = lambda x: (x - [102.9801, 115.9465, 122.7717]) / [1.0, 1.0, 1.0]
    images_mine = torch.tensor(normalizer(np.array(pil_image))).double().permute(2, 0, 1)[None]
    images_mine = images_mine.float().to(device)

    # Extract features
    with torch.no_grad():
        visual_features = feature_network(images_mine)
        visual_features = visual_features.squeeze(0).cpu().numpy()

    # Save features in the specified format
    img_loc_split = img_loc.split('/')
    seq_name = img_loc_split[-1].split('_')[0]  # Extract sequence name (e.g., 'seq1')

    # Save directory structure
    save_dir = os.path.join(
        '/content/drive/MyDrive/vision_and_language_project/Surgical-VQLA/dataset/EndoVis-17-VQLA/vqla/img_features/frcnn'
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save features as .hdf5
    save_path = os.path.join(save_dir, f'{img_loc_split[-1].split(".")[0]}.hdf5')
    with h5py.File(save_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('visual_features', data=visual_features)
    print(f"Saved features to: {save_path}")
