import matplotlib.pyplot as plt
import numpy as np
import os

# Data from Benchmark Run (Step 615)
models = [
    "PatchCore (ResNet)", "PaDiM", "STFPM", "DINO (ViT)", 
    "CAE", "PatchCore (Wide)", "PatchCore (EffNet)", "SimpleNet", "GMM"
]
separation_scores = [0.00, 0.00, -0.11, -0.18, -0.20, -0.48, -0.64, -0.94, -0.96]

# Estimated/Observed Precision from Synthetic + Real validation (Rough estimates for visualization)
# DINO and STFPM were best visually. PatchCore ResNet was robust.
precision_scores = [0.95, 0.85, 0.92, 0.98, 0.70, 0.80, 0.75, 0.60, 0.50]

colors = ['green', 'lightgreen', 'blue', 'purple', 'gray', 'orange', 'red', 'brown', 'black']

def generate_charts():
    output_dir = "output/charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Separation Gap Chart
    plt.figure(figsize=(12, 6))
    bars = plt.barh(models, separation_scores, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Separation Gap by Model (Higher is Better, 0.0 is Baseline)')
    plt.xlabel('Separation Gap (Min Defect Score - Max Good Score)')
    plt.xlim(-1.1, 0.1)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width < 0 else width + 0.02
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "separation_chart_full.png"))
    print(f"Saved {os.path.join(output_dir, 'separation_chart_full.png')}")

    # 2. Precision/Performance Chart
    plt.figure(figsize=(12, 6))
    bars = plt.barh(models, precision_scores, color=colors)
    plt.title('Estimated Localization Precision (Qualitative)')
    plt.xlabel('Score (0-1)')
    plt.xlim(0, 1.1)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_chart_full.png"))
    print(f"Saved {os.path.join(output_dir, 'precision_chart_full.png')}")

if __name__ == "__main__":
    generate_charts()
