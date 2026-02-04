# Gold Chain Defect Detection PoC

Proof of Concept for detecting defects in gold/silver chains using One-Class Anomaly Detection (PatchCore).

## Prerequisites

- Python 3.8+
- [Optional] CUDA-capable GPU

## Installation

1.  Clone the repository or download the files.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Data

Place your **good** (defect-free) training images in:
`dataset/train/good/`

(The internal structure `dataset/train/good` is required).

### 2. Train Model

Run the training command to build the memory bank of normal chain features:

```bash
python main.py train
```
-   This will save the model to `model.pkl`.
-   **Note:** The first run will download the ResNet50 backbone (approx 100MB).

### 3. Inference (Defect Detection)

To check a new image:

```bash
python main.py predict --image_path "path/to/test_image.jpg"
```

**With Jewelry Segmentation (Recommended):**

To analyze only the jewelry and ignore the background:

```bash
python main.py predict --image_path "path/to/test_image.jpg" --segment_jewelry
```

**Output:**
```json
{'anomaly_score': 1.2345, 'prediction': 'OK'}
```

**Optional Visualization:**
To see the anomaly heatmap:
```bash
python main.py predict --image_path "path/to/test_image.jpg" --vis
```

With segmentation visualization:
```bash
python main.py predict --image_path "path/to/test_image.jpg" --segment_jewelry --vis
```

## Features

### Jewelry Segmentation

The `--segment_jewelry` flag enables automatic background removal to focus analysis only on the jewelry:

- **Automatic Detection**: Uses adaptive thresholding and contour detection to identify jewelry
- **Background Removal**: Zeros out background regions in the anomaly map
- **Improved Accuracy**: Prevents false positives from background noise or texture
- **Visualization**: Shows the segmented region when using `--vis`

This feature is especially useful when:
- Images have textured or noisy backgrounds
- Multiple objects are present in the scene
- You want to ensure defects are detected only on the jewelry itself
```

### 4. Configuration

You can adjust parameters in `main.py` or via CLI arguments:
-   `--threshold`: Anomaly score threshold (default: 2.5). Tune this based on your validation data.
-   `--bank_size`: Size of the memory bank (default: 1000). Larger bank = slower but potentially more accurate.
