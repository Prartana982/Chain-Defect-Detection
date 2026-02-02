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

**Output:**
```json
{'anomaly_score': 1.2345, 'prediction': 'OK'}
```

**Optional Visualization:**
To see the anomaly heatmap:
```bash
python main.py predict --image_path "path/to/test_image.jpg" --vis
```

### 4. Configuration

You can adjust parameters in `main.py` or via CLI arguments:
-   `--threshold`: Anomaly score threshold (default: 2.5). Tune this based on your validation data.
-   `--bank_size`: Size of the memory bank (default: 1000). Larger bank = slower but potentially more accurate.
