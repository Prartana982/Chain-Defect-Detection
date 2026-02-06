import os
import shutil
import glob

def organize():
    source_dir = "output_real"
    target_base = "output_of_all_model"
    
    if not os.path.exists(target_base):
        os.makedirs(target_base)
        
    # Process files in output_real: e.g., "img1_DINO.png"
    files = glob.glob(os.path.join(source_dir, "*.png"))
    
    print(f"Moving {len(files)} files from {source_dir} to {target_base}...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Expected format: {image_name}_{ModelName}.png
        # But we need to be careful. E.g. "img1_PatchCore.png" -> split by last underscore?
        
        # logic: split by "_" and take the last part as model name (removing .png)
        # But "PatchCore_Eff" has underscore.
        # Better strategy: Known model names.
        
        known_models = ["DINO", "PatchCore", "STFPM", "PaDiM", "CAE", "SimpleNet", "GMM", "PatchCore_Wide", "PatchCore_Eff"]
        
        model_name = "Unknown"
        for m in known_models:
            if m in filename:
                model_name = m
                break # Take first match? "PatchCore" matches "PatchCore_Eff". 
                      # Should match longest first.
        
        # Sort known_models by length desc to match "PatchCore_Eff" before "PatchCore"
        known_models.sort(key=len, reverse=True)
        
        matched_model = None
        for m in known_models:
            if filename.endswith(f"_{m}.png"):
                matched_model = m
                break
        
        if matched_model:
            # Create model folder
            model_dir = os.path.join(target_base, matched_model)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Move/Copy
            dest_path = os.path.join(model_dir, filename)
            shutil.copy2(file_path, dest_path)
            print(f"Moved {filename} -> {matched_model}/")
        else:
            print(f"Skipping {filename} (Could not identify model)")

    print("Organization complete.")

if __name__ == "__main__":
    organize()
