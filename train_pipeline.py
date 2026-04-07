
######################################################################################
# Configuring importing libraries
######################################################################################
from utils.loader import get_data_loader
from utils.train import fit_model
from utils.eval import test_model
from utils.data_algumentation import ImgTrainTransform, ImgEvalTransform

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sacred import Experiment
from models.models import set_class_model
from sacred.observers import FileStorageObserver
from sentence_transformers import SentenceTransformer
######################################################################################

######################################################################################
# Variables and configurations
######################################################################################

# Getting the local configurations
CLASS_TYPE = "diag" # triage or diag
IMG_TYPE = "clinical" # clinical, dermatoscope or both

with open("./config.json") as json_file:
    _LOCAL_CONFIG = json.load(json_file)

_DATASET_BASE_PATH = _LOCAL_CONFIG["dataset_folder_path"]
_CSV_PATH_TRAIN = os.path.join(_DATASET_BASE_PATH, f"pad-ufes-25-{CLASS_TYPE}_{IMG_TYPE}_folders_raw.csv")
_JSON_PATH_TRAIN = os.path.join(_DATASET_BASE_PATH, f"anamnese_raw_{CLASS_TYPE}_{IMG_TYPE}.json")
_IMGS_FOLDER_TRAIN = os.path.join(_LOCAL_CONFIG['dataset_images_path'])

# Avoiding the tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_COLUMN = "histoMacroCID"
TARGET_NUMBER_COLUMN = "diagnostic-number"
IMG_COLUMN = "img-id"

# Starting sacred experiment
ex = Experiment()

######################################################################################

@ex.config
def cnfg():

    # Defines the folder to be used as validation
    _folder = 1

    # Models configurations
    _use_meta_data = True
    _neurons_reducer_block = 0
    _comb_method = "metablock" # metablock
    _comb_config = [64, 768] # number of metadata
    _batch_size = 30
    _epochs = 50

    _llm_type = "small" # small
    _model_name = 'resnet-50'
    _save_folder = f"results/{CLASS_TYPE}_{_model_name}_{_comb_method}_SE_V2_FITZ__{_llm_type}_folder_{str(_folder)}_{str(time.time()).replace('.', '')}"

    # Training variables
    _best_metric = "loss"
    _pretrained = True
    _optmizer = "adam" # adam or sgd
    _lr_init = 0.0001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None
    _weights = "frequency"       
    

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _lr_init, _sched_factor, _sched_min_lr, _sched_patience, _batch_size, _epochs, 
          _early_stop, _weights, _model_name, _pretrained, _optmizer, _save_folder, _best_metric, _llm_type,
          _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data, _metric_early_stop):
    

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    # Loading the csv file
    csv_all_folders = pd.read_csv(_CSV_PATH_TRAIN)
    
    meta_json = sentence_model = None
    if _use_meta_data:
        with open(_JSON_PATH_TRAIN) as json_file:
            meta_json = json.load(json_file)

        if _llm_type == "small":            
            sentence_model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
        elif _llm_type == "large":
            sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")        
        else:
            raise ValueError("Invalid LLM type")

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    # Loading validation data
    val_imgs_id = val_csv_folder[IMG_COLUMN].values
    val_imgs_path = ["{}/{}".format(_IMGS_FOLDER_TRAIN, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder[TARGET_NUMBER_COLUMN].values
    val_meta_data = list()

    if _use_meta_data:
        for img_id in val_imgs_id:
            doc_vec = sentence_model.encode(meta_json[img_id], show_progress_bar=False)            
            val_meta_data.append(doc_vec)   
        val_meta_data = np.asarray(val_meta_data)
        print(f"-- Using {val_meta_data.shape} meta-data features")        
    else:
        print("-- No metadata")
        val_meta_data = None
    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))
  
  
    print("- Loading training data...")
    train_imgs_id = train_csv_folder[IMG_COLUMN].values
    train_imgs_path = ["{}/{}".format(_IMGS_FOLDER_TRAIN, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder[TARGET_NUMBER_COLUMN].values
    train_meta_data = list()
    if _use_meta_data:
        for img_id in train_imgs_id:
            doc_vec = sentence_model.encode(meta_json[img_id], show_progress_bar=False)           
            train_meta_data.append(doc_vec)     
        train_meta_data = np.asarray(train_meta_data)   
        print(f"-- Using {doc_vec.shape} meta-data features")
    else:
        print("-- No metadata")
        train_meta_data = None

    train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))
    print("-"*50)

    ####################################################################################################################

    ser_lab_freq = train_csv_folder.groupby([TARGET_COLUMN])[IMG_COLUMN].count()
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    print(ser_lab_freq)

    ####################################################################################################################
    print("- Loading", _model_name)

    model = set_class_model(_model_name, len(_labels_name), neurons_reducer_block=_neurons_reducer_block,
                      comb_method=_comb_method, comb_config=_comb_config, pretrained=_pretrained)
    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)

    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())

    if _optmizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=_lr_init)
    elif _optmizer == "sgd":        
        optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    else:
        raise ValueError("Invalid optimizer name")
    
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)
    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model (model, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None, metric_early_stop=_metric_early_stop,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    ####################################################################################################################