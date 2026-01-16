# Deep Learning for Crop Classification using Satellite Image Time Series (SITS)

This repository contains the official implementation code and data for the paper:
"[Insert Your Paper Title Here]"

We propose a robust crop classification framework using Satellite Image Time Series (SITS). This repository includes the preprocessing pipeline, deep learning models (TempCNN, CBAM, ViT), and machine learning baselines (Random Forest, XGBoost).

================================================================================
1. PROJECT STRUCTURE
================================================================================

The repository is organized as follows:

├── dataset/                 # Contains sample datasets (dummy data)
│   ├── sample_nh.csv        # Sample data for region NH
│   ├── sample_nj.csv        # Sample data for region NJ
│   ├── sample_us.csv        # Sample data for region US
│   └── sample_yc.csv        # Sample data for region YC
│
├── weights/                 # Saved model checkpoints (.pt files)
│   ├── CBAM.pt
│   ├── ViT.pt
│   └── ...
│
├── data_augmentation.py     # Data preprocessing & augmentation script
├── DT.ipynb                 # Machine Learning Models (Random Forest, XGBoost)
├── CNN.ipynb                # CNN-based Models (CNN, CBAM, CNN+MLP, CBAM+MLP, CNN+TF, CBAM+TF)
├── TF.ipynb                 # Transformer-based Models (ViT, SSTRE)
├── Clustering.py            # Unsupervised clustering analysis
└── requirements.txt         # List of dependencies

================================================================================
2. REQUIREMENTS
================================================================================

The code is implemented in Python 3.8+ using PyTorch.
To install the necessary dependencies, run the following command:

    pip install -r requirements.txt

* Note: Please install the version of 'torch' compatible with your CUDA environment.
  (Visit https://pytorch.org/ for details).

================================================================================
3. USAGE INSTRUCTIONS
================================================================================

To reproduce the experimental results, please follow the steps below in order.

Step 1: Data Preprocessing
--------------------------
Run the data augmentation script to clean, interpolate, and augment the raw sample data.

    python data_augmentation.py

* Input: dataset/sample_*.csv
* Output: Generates 'train_interpolated.csv' and other split files in the dataset folder.
* Function: Handles missing values, performs linear interpolation for time-series, and balances classes.

Step 2: Model Training & Evaluation
-----------------------------------
We provide Jupyter Notebooks for training different types of models. You can run these notebooks to train models and view evaluation metrics (Accuracy, F1-score, Inference Time).

A. Machine Learning Baselines:
   Open 'DT.ipynb'.
   - This notebook trains Random Forest and XGBoost models.
   - It includes feature engineering and evaluation steps.

B. CNN & Attention Models:
   Open 'CNN.ipynb'.
   - This notebook trains CNN-based Models (CNN, CBAM, CNN+MLP, CBAM+MLP, CNN+TF, CBAM+TF) models.
   - It defines the 1D-CNN architecture and attention mechanisms.

C. Transformer Models:
   Open 'TF.ipynb'.
   - This notebook trains the Transformer-based Models (ViT, SSTRE) adapted for time-series.
   - It includes patch embedding and positional encoding logic.

Each notebook includes:
1. Data Loading
2. Model Architecture Definition
3. Training Loop with Early Stopping
4. Evaluation on Test Set (Accuracy, F1-Score, Cohen's Kappa)
5. Complexity Analysis (Parameters, Inference Time)

================================================================================
4. DATASET AVAILABILITY
================================================================================

Due to privacy and license restrictions regarding the specific agricultural parcels, the full dataset used in the paper cannot be publicly shared.

However, we provide dummy sample datasets ('dataset/sample_*.csv') that preserve the structure and format of the original data. These samples allow reviewers and researchers to verify the reproducibility of the code and pipelines.

* Data Format: Time-series spectral bands (b02, b03, ... b12) collected over 6 months.
* Region Codes: 'nh', 'nj', 'us', 'yc' represent different agricultural regions.

================================================================================
5. LICENSE
================================================================================

This project is licensed under the MIT License.
