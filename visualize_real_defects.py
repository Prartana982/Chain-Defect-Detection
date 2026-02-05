import torch
import cv2
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from src.utils import plot_anomaly
from src.segmentation import segment_jewelry

# Imports (Reusing wrappers)
from train_cae import CAEn
from predict_stfpm import STFPMInferencer
from predict_patchcore import PatchCorePredictor
from predict_padim import PaDiMPredictor

class DINOInferencer:
    def __init__(self, model_path="dino_vits14.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone.to(self.device)
        self.backbone.eval()
        import pickle
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.nbrs = data['nbrs']
    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        t = torch.nn.functional.interpolate(image_tensor, size=((H//14)*14, (W//14)*14), mode='bilinear')
        with torch.no_grad():
            ret = self.backbone.forward_features(t)
            features = ret['x_norm_patchtokens'][0].cpu().numpy()
        distances, _ = self.nbrs.kneighbors(features)
        h, w = t.shape[2]//14, t.shape[3]//14
        amap = distances.reshape(h, w)
        return cv2.resize(amap, (W, H)), np.max(amap)

def visualize_real():
    # Only visualize "img*.jpg" from dataset/test/bad
    test_images = glob.glob("dataset/test/bad/img*.jpg")
    print(f"Found {len(test_images)} real test images.")
    
    if not test_images:
        print("No img*.jpg files found.")
        return

    transform = transforms.Compose([
        transforms.Resize((256, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # All Models
    from predict_stfpm import STFPMInferencer
    from predict_patchcore import PatchCorePredictor
    from predict_padim import PaDiMPredictor
    from predict_simplenet import SimpleNetInferencer
    from predict_patchcore_wide import PatchCoreWideInferencer
    from predict_patchcore_efficient import PatchCoreEffNetInferencer
    from predict_gmm import GMMInferencer
    from train_cae import CAEn
    
    # Simple Wrapper for CAE to match interface
    class CAEInferencer:
        def __init__(self, model_path="cae_model.pth"):
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             self.model = CAEn().to(self.device)
             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
             self.model.eval()
        def predict(self, image_tensor):
            img_tensor = transforms.Compose([transforms.Resize((256, 1024)), transforms.ToTensor()])(image_tensor.squeeze(0)) # Hack: unbatch, resize
            with torch.no_grad(): recon = self.model(img_tensor.unsqueeze(0).to(self.device))
            diff = torch.mean((img_tensor.to(self.device) - recon)**2, dim=1).squeeze().cpu().numpy()
            return diff, np.percentile(diff, 99)

    models_map = {
        "DINO": DINOInferencer("dino_vits14.pkl"),
        "PatchCore": PatchCorePredictor("patchcore_resnet50.pkl"),
        "STFPM": STFPMInferencer("stfpm_model.pth"),
        "PaDiM": PaDiMPredictor("padim_resnet50.pkl"),
        "PatchCore_Wide": PatchCoreWideInferencer("patchcore_wide_model.pkl"),
        "PatchCore_Eff": PatchCoreEffNetInferencer("patchcore_effnet_model.pkl"),
        "SimpleNet": SimpleNetInferencer("simplenet_model.pth"),
        "GMM": GMMInferencer("gmm_model.pkl"),
        "CAE": CAEInferencer("cae_model.pth")
    }
    
    output_dir = "output_real"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    for img_path in test_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {base_name}...")
        
        try:
            image = Image.open(img_path).convert('RGB')
            vis_tensor = transforms.Compose([transforms.Resize((256, 1024)), transforms.ToTensor()])(image)
            _, mask = segment_jewelry(image)
            if mask.sum() == 0: mask = np.ones((256, 1024), dtype=np.uint8)
            else: mask = cv2.resize(mask, (1024, 256), interpolation=cv2.INTER_NEAREST)
            
            input_tensor = transform(image).unsqueeze(0)
            
            for name, model in models_map.items():
                try:
                    ret = model.predict(input_tensor)
                    if isinstance(ret, tuple):
                        amap, score = ret
                    else:
                        amap = ret
                        score = np.max(amap) # Fallback score
                    
                    if amap.shape != (256, 1024): amap = cv2.resize(amap, (1024, 256))
                    
                    masked_map = amap * mask
                    final_score = np.max(masked_map) if mask.sum() > 0 else 0
                    
                    out_path = os.path.join(output_dir, f"{base_name}_{name}.png")
                    # Using 0.8 * max as threshold for vis
                    plot_anomaly(vis_tensor, amap, final_score, final_score*0.8, out_path, mask)
                except Exception as e:
                     print(f"Failed {name} on {base_name}: {e}")

                
        except Exception as e:
            print(f"Error on {img_path}: {e}")

if __name__ == "__main__":
    visualize_real()
