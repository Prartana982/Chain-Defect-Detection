import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

CONFIG = {
    "model_path": "simplenet_model.pth",
    "input_size": (256, 1024),
    "feature_dim": 1536
}

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)

class SimpleNetInferencer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        self.backbone.eval()
        self.model = SimpleNet(CONFIG["feature_dim"]).to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()
        
    def get_features(self, x):
        features = []
        def hook(module, input, output):
            features.append(output)
        h1 = self.backbone.layer2.register_forward_hook(hook)
        h2 = self.backbone.layer3.register_forward_hook(hook)
        _ = self.backbone(x)
        h1.remove()
        h2.remove()
        f1 = features[0]
        f2 = F.interpolate(features[1], size=f1.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([f1, f2], dim=1)

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            features = self.get_features(image_tensor)
        B, C, H, W = features.shape
        features_permuted = features.permute(0, 2, 3, 1).reshape(-1, C)
        with torch.no_grad():
            pred = self.model(features_permuted)
        scores = 1.0 - pred
        algo_map = scores.reshape(B, H, W).cpu().numpy().squeeze()
        algo_map = cv2.resize(algo_map, (CONFIG["input_size"][1], CONFIG["input_size"][0]))
        return algo_map, np.max(algo_map)

if __name__ == "__main__":
    pass
