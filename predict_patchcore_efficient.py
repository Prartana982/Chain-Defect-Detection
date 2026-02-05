import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import cv2
import os

CONFIG = {
    "model_path": "patchcore_effnet_model.pkl",
    "input_size": (256, 1024)
}

class PatchCoreEffNetInferencer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.efficientnet_b4(weights='DEFAULT').to(self.device)
        self.model.eval()
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.transformer = data["transformer"]
                self.knn = data["knn"]
                self.memory_bank = data["memory_bank"]

    def extract_features(self, x):
        outputs = {}
        def hook(name):
            def fn(module, input, output):
                outputs[name] = output
            return fn
        h1 = self.model.features[3].register_forward_hook(hook("stage3"))
        h2 = self.model.features[4].register_forward_hook(hook("stage4"))
        with torch.no_grad():
            _ = self.model(x)
        h1.remove()
        h2.remove()
        f1 = outputs["stage3"]
        f2 = outputs["stage4"]
        f2 = torch.nn.functional.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        feat = torch.cat([f1, f2], dim=1)
        return feat

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        features_map = self.extract_features(image_tensor)
        B, C, H, W = features_map.shape
        features = features_map.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        features = self.transformer.transform(features)
        distances, _ = self.knn.kneighbors(features)
        score_patches = np.mean(distances, axis=1)
        anomaly_map = score_patches.reshape(H, W)
        anomaly_map = cv2.resize(anomaly_map, (CONFIG["input_size"][1], CONFIG["input_size"][0]))
        anomaly_map = cv2.GaussianBlur(anomaly_map, (11, 11), 0)
        return anomaly_map, np.max(anomaly_map)

if __name__ == "__main__":
    pass
