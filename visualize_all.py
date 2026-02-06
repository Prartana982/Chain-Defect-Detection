import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from src.utils import plot_anomaly
from src.segmentation import segment_jewelry

# Imports
from train_cae import CAEn
from predict_simplenet import SimpleNetInferencer
from predict_stfpm import STFPMInferencer
from predict_patchcore_wide import PatchCoreWideInferencer
from predict_patchcore_efficient import PatchCoreEffNetInferencer
from predict_gmm import GMMInferencer
from predict_patchcore import PatchCorePredictor
from predict_padim import PaDiMPredictor

# DINO Wrapper (Local clone)
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

# CAE Wrapper
class CAEInferencer:
    def __init__(self, model_path="cae_model.pth"):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.model = CAEn().to(self.device)
         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
         self.model.eval()
    def predict(self, image_tensor):
        with torch.no_grad(): recon = self.model(image_tensor.to(self.device))
        diff = torch.mean((image_tensor.to(self.device) - recon)**2, dim=1).squeeze().cpu().numpy()
        return diff, np.percentile(diff, 99)

def visualize_all():
    image_path = "dataset/test/bad/Broken.png"
    if not os.path.exists(image_path):
        image_path = "Broken-data1.png" # Root fallback
        
    print(f"Visualizing on {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Segmentation
    _, mask = segment_jewelry(image)
    if mask.sum() == 0: mask = np.ones((256, 1024), dtype=np.uint8)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_cae = transforms.Compose([transforms.Resize((256, 1024)), transforms.ToTensor()])
    
    input_tensor = transform(image).unsqueeze(0)
    input_tensor_cae = transform_cae(image).unsqueeze(0)
    
    vis_tensor = transform_cae(image) # For plotting (0-1 range)
    
    models_map = {
        "SimpleNet": SimpleNetInferencer("simplenet_model.pth"),
        "STFPM": STFPMInferencer("stfpm_model.pth"),
        "PatchCore_Wide": PatchCoreWideInferencer("patchcore_wide_model.pkl"),
        "PatchCore_EffNet": PatchCoreEffNetInferencer("patchcore_effnet_model.pkl"),
        "GMM": GMMInferencer("gmm_model.pkl"),
        "DINO": DINOInferencer("dino_vits14.pkl"),
        "PatchCore": PatchCorePredictor("patchcore_resnet50.pkl"),
        "PaDiM": PaDiMPredictor("padim_resnet50.pkl"),
        "CAE": CAEInferencer("cae_model.pth")
    }
    
    output_dir = "output_all"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    maps = {}
    
    for name, model in models_map.items():
        print(f"Running {name}...")
        try:
            if name == "CAE":
                amap, score = model.predict(input_tensor_cae)
            else:
                amap, score = model.predict(input_tensor)
                
            if amap.shape != (256, 1024): amap = cv2.resize(amap, (1024, 256))
            
            # Mask
            masked_map = amap * cv2.resize(mask, (1024, 256), interpolation=cv2.INTER_NEAREST)
            final_score = np.max(masked_map) if mask.sum() > 0 else 0
            
            # Save
            path = os.path.join(output_dir, f"{name}.png")
            # Heuristic Threshold
            plot_anomaly(vis_tensor, amap, final_score, final_score*0.8, path, mask)
            
            # Store normalized map for ensemble vis
            map_min, map_max = amap.min(), amap.max()
            if map_max > map_min:
                maps[name] = (amap - map_min) / (map_max - map_min)
            else:
                maps[name] = amap
                
        except Exception as e:
            print(f"Failed {name}: {e}")

    # Visualizing Ensemble (Average of DINO, PatchCore, SimpleNet)
    if "DINO" in maps and "PatchCore" in maps and "SimpleNet" in maps:
        print("Generating Ensemble Visualization...")
        ens_map = (maps["DINO"] + maps["PatchCore"] + maps["SimpleNet"]) / 3.0
        path = os.path.join(output_dir, "Ensemble_Top3.png")
        plot_anomaly(vis_tensor, ens_map, np.max(ens_map), 0.5, path, mask)

if __name__ == "__main__":
    visualize_all()
