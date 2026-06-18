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
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


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

def plot_feature_map(feature_tensor, save_path, title="Feature Map"):
    """
    feature_tensor: (C, H, W) or (H, W)
    Multi-channel feature embeddings are visualized by L2 activation magnitude.
    Single-channel logits/probabilities keep their raw spatial response.
    """
    feature_tensor = feature_tensor.detach().float()
    if len(feature_tensor.shape) == 3:
        if feature_tensor.size(0) == 1:
            activation = feature_tensor[0]
        else:
            activation = torch.sqrt(torch.sum(feature_tensor * feature_tensor, dim=0))
    elif len(feature_tensor.shape) == 2:
        activation = feature_tensor
    else:
        raise ValueError("Invalid shape")

    activation = torch.nan_to_num(activation, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
    act_min = activation.min()
    act_max = activation.max()
    if act_max - act_min <= 1e-8:
        activation = np.zeros_like(activation, dtype=np.float32)
    else:
        activation = (activation - act_min) / (act_max - act_min)

    activation = cv2.resize(activation.astype(np.float32), (352, 352), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(np.uint8(np.clip(255 * activation, 0, 255)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(heatmap).save(save_path)
    print(f"Saved: {save_path}")

# Note: Integrate this into test.py or evaluation script where intermediate variables:
# x1, x2, x3, x4 (from backbone)
# high_res_multimasks (from semantic path)
# out1, out2, out (from UNet style detail stream)
# memory_stack (from DMB)
# are available.


def test_logs():
    # Example usage
    path = './test'
    set_log_dir(path)
    print(set_log_dir(path))

if __name__ == "__main__":
    pass
