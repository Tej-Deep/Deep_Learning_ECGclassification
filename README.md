# Deep_Learning_ECGclassification

## Problem Statement

Cardiovascular disease is a leading cause of mortality worldwide, and its diagnosis and management require timely and accurate identification of cardiac abnormalities. Electrocardiogram (ECG) signals provide a non-invasive diagnostic tool for the detection of various cardiac conditions, such as myocardial infarction, conduction disturbances, and hypertrophy. However, the manual interpretation of ECG signals can be time-consuming and subject to inter-observer variability, leading to errors and delayed diagnosis.

Moreover, ECG signals alone may not always provide sufficient information for accurate diagnosis. Clinical notes, which contain information about a patient's medical history, symptoms, and physical examination, can provide valuable context and additional features for accurate diagnosis of cardiac abnormalities.

To address these challenges, this project aims to develop and evaluate deep learning models that leverage both ECG signals and clinical notes for the automated classification of cardiac abnormalities using the PTB-XL dataset. The proposed models will incorporate multi-modal inputs and augment the data using random noise and random masking to improve performance and robustness.

## Dataset Used

The PTB-XL dataset is a publicly available dataset of electrocardiogram (ECG) signals and associated clinical notes. The dataset includes 21799 clinical 12-lead ECGs from 18869 patients of 10 second length. The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. There are in total 71 different ECG statements that conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements. To ensure comparability of machine learning algorithms trained on the dataset, the authors of this dataset also provided recommended splits into training and test sets.

The use of this dataset makes the objective of this project a multi-label, multi-class classification problem because each ECG recording can have multiple labels corresponding to different cardiac abnormalities, and each label corresponds to a specific class. 

The Data set can be downloaded from [here](https://physionet.org/content/ptb-xl/1.0.3/). TO run the training notebooks, download the the dataset and place it under the data folder. YOu also need to unzip the embeddings folder in data to get the embeddings we genereated from the Clinical Notes. The data directory should be simlilar to this after all these actions:

```
- data
    - embeddings
        - 1.pt
        - 2.pt
        ...
    -ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
        - records100/
        - records500/
        - ptbxl_database.csv
        - scp_statements.csv
        ...
```
## Setup

We used Python 3.9.15 in our experiments. To install the required libraries, simply run the following command

### pip
```
pip install -r requirements.txt
```
### conda
```
conda env create -f environment.yml
conda activate DeepLearning
```

## Dataloaders, Checkpoints and Metrics

The results of our experiments and the pretrained  checkpoints can be found [here](https://drive.google.com/drive/folders/1c_zMl6FYbETIaTcHQKVEkOdWEnfaM6Vc)

We have also uploaded .pt files containing the dataloaders so it is alternatively possible to load these .pt files instead of rereading the ECG signal files. To use these dataloader, make sure that the .pt files are in the main directory and set `LOAD_DATASET` to `True` in the notebooks when running and ensure that the `TRAIN_PATH, VALID_PATH,
TEST_PATH` have been set to the correct path.

The .pt files for Metrics contain the training and validation loss over the training period. They can be used to generate a plot by passing the path to the metrics file as an input to the `plot_losses` function in utils/RNN_utils. The below shows how the function can be used.

```
plot_losses(save_dir=*Path to Direcotry of the Metrics file*,
            metrics_save_name=*Metrics file name*)
```

Alternatively, you can plot any metrics save file by using our python script `plot_losses.py` by providing it the save file path as such:

```
python plot_losses.py <metrics file path>
```
