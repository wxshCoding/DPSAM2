import os
from datetime import datetime
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from dataset import TestDataset
from torch.utils.data import DataLoader
import imageio
import shutil
import logging
import cv2
import py_sod_metrics


def init_metrics():
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    # Compatibility across py_sod_metrics versions:
    # some versions require explicit with_dynamic/with_adaptive args.
    try:
        _ = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True)
    except TypeError:
        try:
            _ = py_sod_metrics.MSIoU()
        except TypeError:
            _ = None

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)

    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "fpr": py_sod_metrics.FPRHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
            "ber": py_sod_metrics.BERHandler(**sample_gray),
            "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
            "kappa": py_sod_metrics.KappaHandler(**sample_gray),
            "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
            "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
            "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
            "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
            "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
            "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
            "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
            "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
            "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
            "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
            "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
            "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
            "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
            "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
            "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
            "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
            "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
            "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
            "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
            "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
            "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
            "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
        }
    )
    return FM, WFM, SM, EM, MAE, FMv2

def set_log_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    if not os.path.exists(os.path.join(save_path, timestamp)):
        os.makedirs(os.path.join(save_path, timestamp), exist_ok=True)
    prefix = os.path.join(save_path, timestamp)
    if not os.path.exists(os.path.join(prefix, "logs")):
        os.makedirs(os.path.join(prefix, "logs"), exist_ok=True)
    logs_path = os.path.join(prefix, "logs")
    if not os.path.exists(os.path.join(prefix, "checkpoints")):
        os.makedirs(os.path.join(prefix, "checkpoints"), exist_ok=True)
    checkpoints_path = os.path.join(prefix, "checkpoints")
    return {"log":logs_path,"checkpoints":checkpoints_path}

def create_logger(log_dir, phase='train'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    file = logging.FileHandler(filename=final_log_file)
    # logging.getLogger('').addHandler(console)
    logging.getLogger('').addHandler(file)
    return logger



def test_logs():
    # Example usage
    path = './test'
    set_log_dir(path)
    print(set_log_dir(path))

if __name__ == "__main__":
    pass