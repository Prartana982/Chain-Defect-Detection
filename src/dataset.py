import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class ChainDataset(Dataset):
    def __init__(self, root_dir, resize=256, crop_size=224, segment_jewelry=False):
        self.root_dir = root_dir
        self.segment_jewelry = segment_jewelry
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.[jJ][pP][gG]'), recursive=True) +
                                  glob.glob(os.path.join(root_dir, '**', '*.[pP][nN][gG]'), recursive=True) + 
                                  glob.glob(os.path.join(root_dir, '**', '*.[jJ][pP][eE][gG]'), recursive=True))
        
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.segment_jewelry:
            from src.segmentation import segment_jewelry
            self.segment_fn = segment_jewelry

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply segmentation if enabled
        if self.segment_jewelry:
            image, _ = self.segment_fn(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path
