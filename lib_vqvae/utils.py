import torch
from torch.utils.data import DataLoader
import time
import os
import numpy as np
# import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import pandas as pd


def save_model_and_results(model, eval_adata, eval_loader, results, hyperparameters, timestamp, args):
    # saving model
    SAVE_MODEL_PATH = os.path.join(args.outdir, f'epoch_{timestamp}')
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters}
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/model.pth')