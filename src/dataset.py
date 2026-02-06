import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from src.segmentation import segment_jewelry

class ChainDataset(Dataset):
    def __init__(self, root_dir, transform=None, segment=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            segment (bool): If True, applies segmentation to remove background.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.segment = segment
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.[jJ][pP][gG]'), recursive=True) +
                                  glob.glob(os.path.join(root_dir, '**', '*.[pP][nN][gG]'), recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Determine image size for dummy mask
        w, h = image.size
        
        mask = None
        if self.segment:
            image, mask = segment_jewelry(image)
        
        if self.transform:
            image = self.transform(image)
            
        # Ensure mask is always a tensor for DataLoader collation
        if mask is None:
            # Return dummy mask (1-channel, same spatial dim as image if possible, or just zeros)
            # Since transform might have resized image, we can't easily guess size unless we transform mask too.
            # But here mask is None. Let's return a simple indicator or handle it in training loop.
            # Simplest for default_collate: Return integer 0 if no mask? No, structure must match.
            # Let's return an empty tensor or zeros.
            # Ideally, if we passed 'segment=False', we might not expect a mask.
            # But shared signature demands it.
            # Let's return a small dummy tensor.
            import torch
            mask = torch.zeros((1, 1), dtype=torch.float32)
        else:
             import torch
             if not isinstance(mask, torch.Tensor):
                 mask = torch.from_numpy(mask).unsqueeze(0) # Add channel dim
            
        return image, img_path, mask
