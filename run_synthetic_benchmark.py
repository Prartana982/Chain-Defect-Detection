import torch
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd
import tqdm
from src.segmentation import segment_jewelry

# Import Model wrappers
from train_cae import CAEn
from predict_simplenet import SimpleNetInferencer
from predict_stfpm import STFPMInferencer
from predict_patchcore_wide import PatchCoreWideInferencer
from predict_patchcore_efficient import PatchCoreEffNetInferencer
from predict_gmm import GMMInferencer
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
        H_pad = (H // 14) * 14
        W_pad = (W // 14) * 14
        t = torch.nn.functional.interpolate(image_tensor, size=(H_pad, W_pad), mode='bilinear')
        with torch.no_grad():
            ret = self.backbone.forward_features(t)
            features = ret['x_norm_patchtokens'][0].cpu().numpy()
        distances, _ = self.nbrs.kneighbors(features)
        h, w = t.shape[2]//14, t.shape[3]//14
        amap = distances.reshape(h, w)
        return cv2.resize(amap, (W, H)), np.max(amap)

class CAEInferencer:
    def __init__(self, model_path="cae_model.pth"):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.model = CAEn().to(self.device)
         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
         self.model.eval()
    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad(): recon = self.model(image_tensor)
        diff = torch.mean((image_tensor - recon) ** 2, dim=1).squeeze().cpu().numpy()
        return diff, np.percentile(diff, 99)

def benchmark_synthetic():
    models_map = {
        "PatchCore": PatchCorePredictor("patchcore_resnet50.pkl"),
        "PaDiM": PaDiMPredictor("padim_resnet50.pkl"),
        "CAE": CAEInferencer("cae_model.pth"),
        "DINO": DINOInferencer("dino_vits14.pkl"),
        "SimpleNet": SimpleNetInferencer("simplenet_model.pth"),
        "STFPM": STFPMInferencer("stfpm_model.pth"),
        "PatchCore_Wide": PatchCoreWideInferencer("patchcore_wide_model.pkl"),
        "PatchCore_Eff": PatchCoreEffNetInferencer("patchcore_effnet_model.pkl"),
        "GMM": GMMInferencer("gmm_model.pkl")
    }
    
    synthetic_files = glob.glob("dataset/test/synthetic/*.png")
    if not synthetic_files:
        print("No synthetic files found!")
        return

    transform = transforms.Compose([
        transforms.Resize((256, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_cae = transforms.Compose([transforms.Resize((256, 1024)), transforms.ToTensor()])
    
    results = [] # List of dicts
    
    print("Benchmarking synthetic defects...")
    
    # We need to normalize scores to have fair comparison, but for sensitivity,
    # raw score works if we compare against the model's own baseline (Normal max).
    # Load previously computed "Max Normal" scores? Or recompute.
    # For speed, let's assume we want to see relative strength.
    # We will just print raw scores, or normalized if possible.
    # Let's use Raw Score for now and compare within model across defect types.
    
    for name, model in models_map.items():
        print(f"Testing {name}...")
        
        scores_by_type = {"cut": [], "dent": [], "color": [], "joint": []}
        
        for p in tqdm.tqdm(synthetic_files):
            defect_type = "unknown"
            if "_cut" in p: defect_type = "cut"
            elif "_dent" in p: defect_type = "dent"
            elif "_color" in p: defect_type = "color"
            elif "_joint" in p: defect_type = "joint"
            
            try:
                img = Image.open(p).convert('RGB')
                if name == "CAE":
                    t = transform_cae(img).unsqueeze(0)
                else:
                    t = transform(img).unsqueeze(0)
                    
                _, score = model.predict(t)
                scores_by_type[defect_type].append(float(score))
            except:
                pass
                
        # Aggregate
        row = {"Model": name}
        for dtype, vals in scores_by_type.items():
            if vals:
                row[dtype] = np.mean(vals)
            else:
                row[dtype] = 0.0
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Normalize per column to 0-1 to see relative "Sensitivity"
    # because PaDiM scores are ~20, PatchCore ~5, DINO ~10
    df_norm = df.copy()
    for col in ["cut", "dent", "color", "joint"]:
        max_val = df[col].max()
        if max_val > 0:
            df_norm[col] = df[col] / max_val
            
    print("\nSensitivity Analysis (Normalized 0-1):")
    print(df_norm)
    df_norm.to_csv("synthetic_sensitivity_results.csv")

if __name__ == "__main__":
    benchmark_synthetic()
