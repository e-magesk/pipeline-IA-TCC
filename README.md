# Deep Learning Pipeline for Skin Cancer Detection

This repository contains a from-scratch PyTorch deep learning pipeline designed for skin lesion classification. The pipeline handles everything from raw data preprocessing to model training, incorporating both images and clinical metadata (anamnesis).

## 🛠️ Environment Setup

To ensure reproducibility, this project uses a Conda environment containing all the necessary dependencies (PyTorch, CUDA support, and data processing libraries).

1. Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
2. Clone this repository and navigate to the project root.
3. Create the environment using the provided `ai-environment.yml` file:

    ```bash
    conda env create -f ai-environment.yml
    ```

4. Activate the environment:

    ```bash
    conda activate ai-environment
    ```

## ⚙️ Project Configuration

Before running any scripts or notebooks, you must define the local paths to your dataset. 

Create a file named `config.json` in the **root directory** of the project with the following structure. Update the values with the absolute or relative paths to your local machine's data:

```json
{
    "dataset_folder_path": "/path/to/your/base/dataset/folder",
    "dataset_csv_path": "/path/to/your/anonymous-metadata.csv",
    "dataset_images_path": "/path/to/your/images/folder"
}
```

## 🧹 Data Preprocessing

The first step in the pipeline is to clean, filter, and organize the raw dataset. This is handled interactively in the `preprocess/preprocess_data.ipynb` notebook.

The preprocessing stage is responsible for:

* **Image Filtering:** Selecting specific image acquisition types (e.g., Clinical vs. Dermatoscope).
* **Diagnostic Clustering:** Mapping raw ICD-10 codes into broader, model-friendly target classes (e.g., Melanoma, Basal Cell Carcinoma).
* **Metadata Consolidation:** Extracting patient clinical features (age, skin type, lesion history) and formatting them into a standardized anamnesis dictionary for multimodal training.
* **Stratified K-Fold Splitting:** Dividing the dataset into 5 training/validation folds to ensure robust model evaluation.

Running this notebook will generate the cleaned CSV files and the `anamnese_raw.json` required for the subsequent PyTorch `Dataset` instantiation.