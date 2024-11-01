import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def seeding(seed):
    """
    Sets the seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """
    Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    """
    Calculates the time taken for each epoch.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics such as Jaccard, F1 score, recall, precision, and accuracy.
    """
    y_true = y_true.cpu().numpy().astype(np.uint8).reshape(-1)
    y_pred = y_pred.cpu().numpy().astype(np.uint8).reshape(-1)

    jaccard = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return [jaccard, f1, recall, precision, accuracy]

def mask_parse(mask):
    """
    Parses a binary mask for visualization.
    """
    mask = np.expand_dims(mask, axis=-1)
    return np.concatenate([mask, mask, mask], axis=-1)
