import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
import cv2
import os

CONFIG = {
    "model_path": "stfpm_model.pth",
    "input_size": (256, 1024)
}

class STFPMInferencer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        self.teacher.eval()
        self.student = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.student.load_state_dict(state)
        self.student.eval()
        
    def extract_features(self, model, x):
        features = []
        def hook(module, input, output):
            features.append(output)
        h1 = model.layer1.register_forward_hook(hook)
        h2 = model.layer2.register_forward_hook(hook)
        h3 = model.layer3.register_forward_hook(hook)
        _ = model(x)
        h1.remove()
        h2.remove()
        h3.remove()
        return features

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            t_feats = self.extract_features(self.teacher, image_tensor)
            s_feats = self.extract_features(self.student, image_tensor)
            
        anomaly_map = torch.zeros((image_tensor.shape[2], image_tensor.shape[3])).to(self.device)
        for t, s in zip(t_feats, s_feats):
            diff = torch.sum((t - s) ** 2, dim=1, keepdim=True)
            diff_resized = F.interpolate(diff, size=(image_tensor.shape[2], image_tensor.shape[3]), mode='bilinear', align_corners=False)
            anomaly_map += diff_resized.squeeze()
            
        result_map = anomaly_map.cpu().numpy()
        result_map = cv2.GaussianBlur(result_map, (11, 11), 0)
        return result_map, np.max(result_map)

if __name__ == "__main__":
    pass
