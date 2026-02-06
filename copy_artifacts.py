import os
import shutil

def copy_artifacts():
    # Source Directories
    base_dir = r"c:\Users\Jaspreet\Desktop\chain\Chain-Defect-Detection"
    chart_dir = os.path.join(base_dir, "output", "charts")
    all_models_dir = os.path.join(base_dir, "output_of_all_model")
    
    # Destination Artifact Directory
    dest_dir = r"C:\Users\Jaspreet\.gemini\antigravity\brain\1c6a5dcd-695b-4630-a585-2bbf3bbd5f1f"
    
    # 1. Copy Charts
    charts = ["separation_chart_full.png", "precision_chart_full.png"]
    for chart in charts:
        src = os.path.join(chart_dir, chart)
        if os.path.exists(src):
            shutil.copy2(src, dest_dir)
            print(f"Copied {chart}")
        else:
            print(f"MISSING Chart: {src}")

    # 2. Copy Real Validation Image Sets (DINO, PatchCore, STFPM)
    # We want img1 to img6
    models = ["DINO", "PatchCore", "STFPM"] 
    # PaDiM failed, so skipping. GMM exists if we want it.
    
    for model in models:
        model_dir = os.path.join(all_models_dir, model)
        if not os.path.exists(model_dir):
            print(f"MISSING Model Dir: {model_dir}")
            continue
            
        for i in range(1, 7):
            img_name = f"img{i}_{model}.png"
            src = os.path.join(model_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dest_dir)
                print(f"Copied {img_name}")
            else:
                 # Try approximate name? organize_outputs.py used simple logic
                 print(f"MISSING: {src}")

if __name__ == "__main__":
    copy_artifacts()
