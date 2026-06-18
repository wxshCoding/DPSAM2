import os
import subprocess
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset_new, FullDataset_new_bbox, collate_fn_multi_points, collate_fn_bbox
from mmsam2 import MMSAM2
import _utils as ff

# ------------------------------------------------------------------ #
#  Task configuration registry                                         #
#  Usage: python train.py --task Polyp                                 #
#         python train.py --task Marine                                #
#         python train.py --task Camouflaged                           #
#         python train.py --task Salient                               #
# ------------------------------------------------------------------ #
# 参数功能说明:
# 1) 任务与数据路径
#    --task: 选择预设任务类型，会自动设置 data_path、valid_list 和 eval_script。
#    --exp_name: 实验名称，用于 logs/<exp_name>/... 的日志、checkpoint、预测结果目录。
#    --data_path: 数据集根目录；如果显式传入，会覆盖 --task 中的默认数据路径。
#    --valid_list: 验证/测试子集名称列表；如果显式传入，会覆盖 --task 中的默认验证集。
#    --eval_script: 外部评估脚本路径；当前训练主流程以内置 evaluate_valid_sets 为主。
# 2) 模型与恢复训练
#    --hiera_path: SAM2 Hiera 预训练权重路径，用于初始化 MMSAM2 内部的 SAM2 主干。
#    --resume_checkpoint/--checkpoint: 恢复训练用 checkpoint；会加载模型权重、optimizer、scheduler
#        和保存的 epoch。DMB 记忆库默认清空重建，只有传入 --resume_memory_bank 才从 checkpoint 恢复。
#    --resume_memory_bank: 显式恢复 checkpoint 中的 DMB；如果上次保存的 DMB 已污染，不要开启。
#    --save_path: 日志、checkpoint、验证预测结果的根目录。
# 3) 训练日程与优化
#    --epoch: 总训练 epoch 数；resume 时从 checkpoint 记录的下一轮继续直到该总数。
#    --valid_interval: 每隔多少个 epoch 进行一次内部验证，同时保存 checkpoint。
#    --lr: AdamW 初始学习率。
#    --batch_size: 训练 batch size；若最后一批只剩 1 张图，会自动 drop_last 以避免 BatchNorm 报错。
#    --weight_decay: AdamW 权重衰减。
#    --seed: 随机种子；传入后固定 Python、NumPy、PyTorch 随机性，便于复现实验。
# 4) Prompt 设置
#    --train_prompt_type: 训练使用的交互提示类型。point 表示前景点提示，bbox 表示边界框提示。
#        训练时还会以 50% 概率执行 prompt dropout，即 current_click=None，用于增强无提示鲁棒性。
# 5) 评估指标参数
#    --boundary_iou_ratio: Boundary IoU 的边界带宽，按图像对角线比例计算。
#    --boundary_f_ratio: Boundary F-score 的边界匹配容忍距离，按图像对角线比例计算。
#    --nsd_ratio: Normalized Surface Dice 的表面距离容忍阈值，按图像对角线比例计算。
# 6) 损失权重
#    --sam_aux_loss_weight: SAM2 semantic stream 的辅助监督权重，对 high_res_multimasks 单独计算 loss，
#        用于避免 SAM2 分支在融合训练中被 UNet-style decoder 掩盖。
#    --unet_side_loss_weight: UNet-style decoder 两个 side output 的辅助监督权重；默认降低到 0.3，
#        让最终融合输出和 SAM2 分支获得更直接的优化压力。
# 7) DMB memory 写入策略
#    --memory_mask_source: 控制 _encode_new_memory 使用哪一路 mask logits 编码 DMB memory。
#        sam: 使用 SAM2 high_res_multimasks；fused: 使用最终融合预测 pr；blend: 使用 pr 与
#        high_res_multimasks 的加权混合。默认 fused，使记忆库写入与最终预测目标一致。
#    --memory_mask_blend: 当 --memory_mask_source=blend 时，pr 的混合权重；默认 0.7。
# 8) 输出与可视化
#    --save_predictions: 开启后保存内部验证预测，包括融合预测 logits/prob/16-bit png，以及
#        high_res_multimasks 单独预测图。
#    --save_feature_vis: 开启后保存特征热力图，包括 MFB/DMB/SAM2 semantic/UNet decoder/epoch-level DMB。
#    --feature_vis_dir: 特征热力图保存根目录。
TASK_CONFIGS = {
    "Polyp": {
        "data_path":   "../data/Polyp",
        # "valid_list":  ["CVC-ColonDB", "Kvasir", "ETIS-LaribPolypDB", "CVC-300", "CVC-ClinicDB"],
        "valid_list":  ["Kvasir", "ETIS-LaribPolypDB", "CVC-300", "CVC-ClinicDB"],
        "eval_script": "./polyp_auto.sh",
    },
    "Marine": {
        "data_path":   "../data/Marine",
        "valid_list":  ["MAS3K", "RMAS"],
        "eval_script": "./marine_auto.sh",
    },
    "Camouflaged": {
        "data_path":   "../data/Camouflaged",
        "valid_list":  ["CAMO", "CHAMELEON", "COD10K", "NC4K"],
        "eval_script": "./camouflaged_auto.sh",
    },
    "Salient": {
        "data_path":   "../data/Salient",
        "valid_list":  ["DUT-OMRON", "DUTS-TE", "ECSSD", "HKU-IS", "PASCAL-S"],
        "eval_script": "./salient_auto.sh",
    },
}

parser = argparse.ArgumentParser(
    "mmsam2 training",
    formatter_class=argparse.RawTextHelpFormatter,
)

# Fine-grained overrides (all optional when --task is given)
parser.add_argument("--exp_name",    type=str,  default=None, help="Experiment name (defaults to --task value)")
parser.add_argument("--data_path",   type=str,  default=None, help="Path to dataset root (overrides task config)")
parser.add_argument("--valid_list",  nargs='+', default=None, help="Validation subset names (overrides task config)")
parser.add_argument("--eval_script", type=str,  default=None, help="Path to eval shell script (overrides task config)")

parser.add_argument("--hiera_path",  type=str,  default="../data/sam2.pt", help="Path to SAM2 pretrained checkpoint")
parser.add_argument("--save_path",   type=str,  default="./logs", help="Directory to store checkpoints and logs")
parser.add_argument(
    "--task", type=str, default='Polyp',
    choices=list(TASK_CONFIGS.keys()),
    help=(
        "Task name — auto-configures data_path, valid_list and eval_script.\n"
        "Available: " + ", ".join(TASK_CONFIGS.keys()) + "\n"
        "Example:   python train.py --task Polyp"
    ),
)
# parser.add_argument(
#     "--task", type=str, default='Camouflaged',
#     choices=list(TASK_CONFIGS.keys()),
#     help=(
#         "Task name — auto-configures data_path, valid_list and eval_script.\n"
#         "Available: " + ", ".join(TASK_CONFIGS.keys()) + "\n"
#         "Example:   python train.py --task Camouflaged"
#     ),
# )
# parser.add_argument(
#     "--task", type=str, default='Marine',
#     choices=list(TASK_CONFIGS.keys()),
#     help=(
#         "Task name — auto-configures data_path, valid_list and eval_script.\n"
#         "Available: " + ", ".join(TASK_CONFIGS.keys()) + "\n"
#         "Example:   python train.py --task Marine"
#     ),
# )
parser.add_argument(
    "--resume_checkpoint", "--checkpoint",
    dest="resume_checkpoint",
    type=str,
    default="./checkpoints/polyp_184_2026_05_30_130702important.pth",
    # default="logs/Polyp/2026_06_10_184930/checkpoints/polyp_206_2026_06_10_194732.pth",
    help="Resume training from checkpoint path (loads model and, when available, optimizer/scheduler/epoch).",
)
parser.add_argument(
    "--resume_memory_bank",
    action="store_true",
    help="Also restore DMB memory_bank_state from checkpoint. Leave off to rebuild DMB from current training data.",
)
# parser.add_argument(
#     "--resume_checkpoint", "--checkpoint",
#     dest="resume_checkpoint",
#     type=str,
#     default="./checkpoints/Marine.pth",
#     help="Resume training from checkpoint path (loads model/memory bank and, when available, optimizer/scheduler/epoch).",
# )
# parser.add_argument(
#     "--resume_checkpoint", "--checkpoint",
#     dest="resume_checkpoint",
#     type=str,
#     default="./checkpoints/camouflaged.pth",
#     help="Resume training from checkpoint path (loads model/memory bank and, when available, optimizer/scheduler/epoch).",
# )
# parser.add_argument(
#     "--resume_checkpoint", "--checkpoint",
#     dest="resume_checkpoint",
#     type=str,
#     # default="./checkpoint/marine_16_2026_05_31_142751.pth",
#     default="./checkpoint/Marine.pth",
#     help="Resume training from checkpoint path (loads model/memory bank and, when available, optimizer/scheduler/epoch).",
# )
parser.add_argument("--epoch",       type=int,  default=300,  help="Number of training epochs")
parser.add_argument("--valid_interval", type=int, default=1, help="Run validation every N epochs")
parser.add_argument("--lr",          type=float, default=0.001, help="Learning rate")
# parser.add_argument("--batch_size",  type=int,  default=8)
parser.add_argument("--batch_size",  type=int,  default=5)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--seed",        type=int, default=None, help="Random seed for reproducible multi-seed runs")
parser.add_argument(
    "--train_prompt_type",
    type=str,
    default="bbox",
    choices=["point", "bbox"],
    help="Prompt type used during training: point or bbox.",
)
parser.add_argument("--boundary_iou_ratio", type=float, default=0.02, help="Boundary IoU band width as ratio of image diagonal")
parser.add_argument("--boundary_f_ratio", type=float, default=0.008, help="Boundary F-score tolerance as ratio of image diagonal")
parser.add_argument("--nsd_ratio", type=float, default=0.008, help="Normalized surface Dice tolerance as ratio of image diagonal")
parser.add_argument(
    "--sam_aux_loss_weight",
    type=float,
    default=1.0,
    help="Weight for auxiliary supervision on SAM2 high_res_multimasks.",
)
parser.add_argument(
    "--unet_side_loss_weight",
    type=float,
    default=0.3,
    help="Weight for each UNet-style side-output auxiliary loss.",
)
parser.add_argument(
    "--memory_mask_source",
    type=str,
    default="fused",
    choices=["sam", "fused", "blend"],
    help="Mask logits source used by SAM2 memory encoder: sam=high_res_multimasks, fused=pr, blend=weighted mix.",
)
parser.add_argument(
    "--memory_mask_blend",
    type=float,
    default=0.7,
    help="Blend weight for pr when --memory_mask_source=blend.",
)
parser.add_argument(
    "--etis_boundary_optimized_fusion",
    action="store_true",
    help="Enable conservative ETIS-oriented residual fusion that suppresses detail residuals opposing confident SAM2 logits.",
)
parser.add_argument(
    "--etis_opposite_residual_scale",
    type=float,
    default=0.35,
    help="Residual scale used by --etis_boundary_optimized_fusion when detail residual opposes confident SAM2 logits.",
)
parser.add_argument(
    "--etis_confidence_margin",
    type=float,
    default=4.0,
    help="SAM logit magnitude at which ETIS opposite-residual suppression reaches its full strength.",
)
parser.add_argument( # 预测图生成
    "--save_predictions",
    action="store_true",
    help="Save validation prediction outputs during internal evaluation.",
)
parser.add_argument(
    "--save_feature_vis",
    action="store_true",
    help="Save MFB, retrieved DMB, SAM2 semantic stream, U-Net decoder, and epoch-level DMB memory feature maps during validation.",
)
parser.add_argument(
    "--feature_vis_dir",
    type=str,
    default="feature_vis",
    help="Directory used for saved feature visualization maps.",
)
args = parser.parse_args()

# Resolve task config — CLI overrides take precedence over task defaults
if args.task is not None:
    cfg = TASK_CONFIGS[args.task]
    if args.exp_name    is None: args.exp_name    = args.task
    if args.data_path   is None: args.data_path   = cfg["data_path"]
    if args.valid_list  is None: args.valid_list  = cfg["valid_list"]
    if args.eval_script is None: args.eval_script = cfg["eval_script"]
else:
    # Backward-compatible defaults (Polyp) when no --task is given
    if args.exp_name    is None: args.exp_name    = "Polyp"
    if args.data_path   is None: args.data_path   = "../data/Polyp"
    if args.valid_list  is None: args.valid_list  = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    if args.eval_script is None: args.eval_script = "./polyp_auto.sh"

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def intersectionAndUnion(imPred, imLab, numClass=1):
    # Use binary labels {0, 1}; count foreground class statistics.
    imPred = imPred.astype(np.uint8)
    imLab = imLab.astype(np.uint8)
    hist_range = (1, numClass + 1)

    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=hist_range)
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=hist_range)
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=hist_range)
    area_union = area_pred + area_lab - area_intersection
    area_sum = area_pred + area_lab

    return area_intersection, area_union, area_sum

def _mask_to_boundary(mask, dilation):
    """Convert a binary mask to boundary map by erosion subtraction."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    dilation = max(1, int(dilation))
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Pad to preserve image-border boundaries.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, kernel, iterations=dilation)
    boundary = padded - eroded
    return boundary[1:-1, 1:-1]

def _boundary_iou(pred_mask, gt_mask, ratio=0.02):
    h, w = pred_mask.shape
    boundary_width = max(1, int(round(ratio * np.sqrt(h * h + w * w))))
    pred_boundary = _mask_to_boundary(pred_mask, boundary_width)
    gt_boundary = _mask_to_boundary(gt_mask, boundary_width)

    intersection = np.logical_and(pred_boundary > 0, gt_boundary > 0).sum()
    union = np.logical_or(pred_boundary > 0, gt_boundary > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-8))

def _boundary_f_score(pred_mask, gt_mask, ratio=0.008):
    h, w = pred_mask.shape
    tolerance = max(1, int(round(ratio * np.sqrt(h * h + w * w))))
    kernel = np.ones((3, 3), dtype=np.uint8)

    pred_boundary = _mask_to_boundary(pred_mask, 1)
    gt_boundary = _mask_to_boundary(gt_mask, 1)

    pred_count = int((pred_boundary > 0).sum())
    gt_count = int((gt_boundary > 0).sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    pred_match = cv2.dilate(pred_boundary, kernel, iterations=tolerance)
    gt_match = cv2.dilate(gt_boundary, kernel, iterations=tolerance)

    precision = np.logical_and(pred_boundary > 0, gt_match > 0).sum() / (pred_count + 1e-8)
    recall = np.logical_and(gt_boundary > 0, pred_match > 0).sum() / (gt_count + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def _hd_hd95(pred_mask, gt_mask):
    """
    Symmetric Hausdorff distance / 95th percentile Hausdorff distance
    computed on mask boundaries (pixel space).
    """
    h, w = pred_mask.shape
    diagonal = float(np.sqrt(h * h + w * w))
    pred_boundary = _mask_to_boundary(pred_mask, 1) > 0
    gt_boundary = _mask_to_boundary(gt_mask, 1) > 0

    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())

    if pred_count == 0 and gt_count == 0:
        return 0.0, 0.0
    if pred_count == 0 or gt_count == 0:
        return diagonal, diagonal

    mask_precise = cv2.DIST_MASK_PRECISE if hasattr(cv2, "DIST_MASK_PRECISE") else 3
    pred_to_gt_map = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, mask_precise)
    gt_to_pred_map = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, mask_precise)

    d_pred_to_gt = pred_to_gt_map[pred_boundary]
    d_gt_to_pred = gt_to_pred_map[gt_boundary]

    hd = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))
    hd95 = float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))
    return hd, hd95

def _normalized_surface_dice(pred_mask, gt_mask, ratio=0.008):
    """
    Normalized surface Dice in pixel space.
    A surface point is counted as matched when its Euclidean distance to the
    opposite surface is within ratio * image diagonal.
    """
    h, w = pred_mask.shape
    tolerance = max(1, int(round(ratio * np.sqrt(h * h + w * w))))
    pred_boundary = _mask_to_boundary(pred_mask, 1) > 0
    gt_boundary = _mask_to_boundary(gt_mask, 1) > 0

    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    mask_precise = cv2.DIST_MASK_PRECISE if hasattr(cv2, "DIST_MASK_PRECISE") else 3
    pred_to_gt_map = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, mask_precise)
    gt_to_pred_map = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, mask_precise)

    pred_match = int((pred_to_gt_map[pred_boundary] <= tolerance).sum())
    gt_match = int((gt_to_pred_map[gt_boundary] <= tolerance).sum())
    return float((pred_match + gt_match) / (pred_count + gt_count + 1e-8))

def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(finite_values.mean())


FUSION_STAT_NAMES = [
    "fusion_T1",
    "fusion_T2",
    "fusion_detail_gain",
    "fusion_detail_gate_mean",
    "fusion_boundary_gate_mean",
    "fusion_etis_opt",
    "fusion_opposite_scale_mean",
    "fusion_opposite_ratio",
    "fusion_sam_abs_mean",
    "fusion_unet_abs_mean",
    "fusion_fused_abs_mean",
    "fusion_unet_to_sam_abs",
    "fusion_same_sign",
    "fusion_conflict",
    "fusion_pr_sam_agree",
    "fusion_pr_unet_agree",
    "fusion_sam_fg",
    "fusion_unet_fg",
    "fusion_pr_fg",
    "fusion_sam_keep",
    "fusion_unet_keep",
    "fusion_sam_removed",
]


def _empty_fusion_stats():
    return {name: 0.0 for name in FUSION_STAT_NAMES}


def _fusion_hint(metrics):
    pr_sam_agree = metrics["fusion_pr_sam_agree"]
    pr_unet_agree = metrics["fusion_pr_unet_agree"]
    unet_to_sam = metrics["fusion_unet_to_sam_abs"]
    sam_removed = metrics["fusion_sam_removed"]

    if sam_removed > 0.5 and pr_unet_agree >= pr_sam_agree:
        return "unet_veto_sam"
    if pr_unet_agree > pr_sam_agree + 0.05 and unet_to_sam >= 1.1:
        return "unet_dominant"
    if pr_sam_agree > pr_unet_agree + 0.05 and unet_to_sam <= 0.9:
        return "sam_dominant"
    return "mixed"


def _format_fusion_stats(metrics):
    return (
        "hint={hint}, T1(unused)={fusion_T1:.4f}, T2(unused)={fusion_T2:.4f}, "
        "detail_gain={fusion_detail_gain:.4f}, detail_gate={fusion_detail_gate_mean:.4f}, "
        "boundary_gate={fusion_boundary_gate_mean:.4f}, etis_opt={fusion_etis_opt:.0f}, "
        "opp_scale={fusion_opposite_scale_mean:.4f}, opp_ratio={fusion_opposite_ratio:.4f}, "
        "sam_abs={fusion_sam_abs_mean:.4f}, unet_abs={fusion_unet_abs_mean:.4f}, "
        "pr_abs={fusion_fused_abs_mean:.4f}, unet/sam={fusion_unet_to_sam_abs:.4f}, "
        "same_sign={fusion_same_sign:.4f}, conflict={fusion_conflict:.4f}, "
        "pr~sam={fusion_pr_sam_agree:.4f}, pr~unet={fusion_pr_unet_agree:.4f}, "
        "fg[sam/unet/pr]={fusion_sam_fg:.4f}/{fusion_unet_fg:.4f}/{fusion_pr_fg:.4f}, "
        "sam_keep={fusion_sam_keep:.4f}, unet_keep={fusion_unet_keep:.4f}, "
        "sam_removed={fusion_sam_removed:.4f}"
    ).format(hint=_fusion_hint(metrics), **metrics)


def _safe_path_name(value):
    value = str(value)
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def _get_eval_sample_name(dataloader, sample_idx, batch_item_idx):
    dataset = dataloader.dataset
    if hasattr(dataset, "images"):
        absolute_idx = sample_idx + batch_item_idx
        if absolute_idx < len(dataset.images):
            return os.path.splitext(os.path.basename(dataset.images[absolute_idx]))[0]
    return f"sample_{sample_idx + batch_item_idx:06d}"


def _save_prediction_outputs(save_dir, sample_name, pred_logit, pred_prob, high_res_multimask_prob=None):
    logits_dir = os.path.join(save_dir, "logits_npy")
    prob_dir = os.path.join(save_dir, "prob_npy")
    preview_dir = os.path.join(save_dir, "prob_u16_png")
    high_res_multimasks_dir = os.path.join(save_dir, "high_res_multimasks")
    os.makedirs(logits_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    if high_res_multimask_prob is not None:
        os.makedirs(high_res_multimasks_dir, exist_ok=True)

    safe_name = _safe_path_name(sample_name)
    np.save(os.path.join(logits_dir, f"{safe_name}.npy"), pred_logit.astype(np.float32, copy=False))
    np.save(os.path.join(prob_dir, f"{safe_name}.npy"), pred_prob.astype(np.float32, copy=False))

    prob_u16 = np.rint(np.clip(pred_prob, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(os.path.join(preview_dir, f"{safe_name}.png"), prob_u16)

    if high_res_multimask_prob is not None:
        high_res_u16 = np.rint(np.clip(high_res_multimask_prob, 0.0, 1.0) * 65535.0).astype(np.uint16)
        cv2.imwrite(os.path.join(high_res_multimasks_dir, f"{safe_name}.png"), high_res_u16)


def _reset_memory_bank(model):
    model.memory_bank.memories = []
    model.memory_bank.usage_counts = []
    model.memory_bank.timestamps = []
    model.memory_bank.current_time = 0


def _is_valid_memory_item(memory):
    if not isinstance(memory, (list, tuple)) or len(memory) != 4:
        return False
    feature, pos_enc, iou_score, image_embed = memory
    tensors = [feature, pos_enc, image_embed]
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        return False
    if feature.numel() == 0 or pos_enc.numel() == 0 or image_embed.numel() == 0:
        return False
    if not all(torch.isfinite(item.detach()).all().item() for item in tensors):
        return False
    if isinstance(iou_score, torch.Tensor):
        if iou_score.numel() == 0 or not torch.isfinite(iou_score.detach()).all().item():
            return False
    return True


def _memory_bank_state(model):
    valid_memories = []
    valid_usage_counts = []
    valid_timestamps = []
    for idx, memory in enumerate(model.memory_bank.memories):
        if not _is_valid_memory_item(memory):
            continue
        valid_memories.append(memory)
        if idx < len(model.memory_bank.usage_counts):
            valid_usage_counts.append(model.memory_bank.usage_counts[idx])
        else:
            valid_usage_counts.append(1)
        if idx < len(model.memory_bank.timestamps):
            valid_timestamps.append(model.memory_bank.timestamps[idx])
        else:
            valid_timestamps.append(model.memory_bank.current_time)

    return {
        'memories': valid_memories,
        'max_size': model.memory_bank.max_size,
        'min_size': model.memory_bank.min_size,
        'similarity_threshold': model.memory_bank.similarity_threshold,
        'decay_factor': model.memory_bank.decay_factor,
        'usage_counts': valid_usage_counts,
        'timestamps': valid_timestamps,
        'current_time': model.memory_bank.current_time
    }


def _restore_memory_bank(model, memory_state, device, logger=None):
    log_warn = logger.warning if logger is not None else print
    memories = memory_state.get("memories", [])
    usage_counts = memory_state.get("usage_counts", [])
    timestamps = memory_state.get("timestamps", [])

    device_memories = []
    device_usage_counts = []
    device_timestamps = []
    dropped = 0
    for idx, memory in enumerate(memories):
        if not _is_valid_memory_item(memory):
            dropped += 1
            continue
        device_memory = []
        for item in memory:
            if isinstance(item, torch.Tensor):
                device_memory.append(torch.nan_to_num(item.to(device), nan=0.0, posinf=0.0, neginf=0.0).detach())
            else:
                device_memory.append(item)
        device_memories.append(device_memory)
        device_usage_counts.append(usage_counts[idx] if idx < len(usage_counts) else 1)
        device_timestamps.append(timestamps[idx] if idx < len(timestamps) else 0)

    model.memory_bank.memories = device_memories
    if dropped > 0:
        log_warn(f"[Resume] dropped invalid DMB memories: {dropped} / {len(memories)}")

    # model.memory_bank.max_size = memory_state.get("max_size", model.memory_bank.max_size)
    # model.memory_bank.min_size = memory_state.get("min_size", model.memory_bank.min_size)
    model.memory_bank.min_size = 2
    model.memory_bank.max_size = 4

    # model.memory_bank.similarity_threshold = memory_state.get(
    #     "similarity_threshold", model.memory_bank.similarity_threshold
    # )
    model.memory_bank.similarity_threshold = 0.45
    model.memory_bank.decay_factor = memory_state.get("decay_factor", model.memory_bank.decay_factor)
    model.memory_bank.usage_counts = device_usage_counts
    model.memory_bank.timestamps = device_timestamps
    model.memory_bank.current_time = memory_state.get("current_time", 0)


def load_training_checkpoint(model, checkpoint_path, device, optimizer=None, scheduler=None, logger=None, resume_memory_bank=False):
    log = logger.info if logger is not None else print
    log_warn = logger.warning if logger is not None else print

    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = 0

    # 1) Model weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    if missing_keys:
        log_warn(f"[Resume] missing keys: {len(missing_keys)}")
    if unexpected_keys:
        log_warn(f"[Resume] unexpected keys: {len(unexpected_keys)}")

    # 2) Memory bank
    if resume_memory_bank and isinstance(checkpoint, dict) and "memory_bank_state" in checkpoint:
        _restore_memory_bank(model, checkpoint["memory_bank_state"], device, logger=logger)
        log(f"[Resume] memory bank state loaded: {len(model.memory_bank.memories)} memories.")
    elif resume_memory_bank:
        _reset_memory_bank(model)
        log_warn("[Resume] no memory bank state found; memory bank is reset.")
    else:
        _reset_memory_bank(model)
        log("[Resume] memory bank reset; pass --resume_memory_bank to restore DMB state.")

    # 3) Optimizer / scheduler (optional compatibility)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log("[Resume] optimizer state loaded.")
    elif optimizer is not None:
        log_warn("[Resume] optimizer state not found in checkpoint; using fresh optimizer.")

    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log("[Resume] scheduler state loaded.")
    elif scheduler is not None:
        log_warn("[Resume] scheduler state not found in checkpoint; using fresh scheduler.")

    # 4) Start epoch
    if isinstance(checkpoint, dict):
        saved_epoch = checkpoint.get("epoch", -1)
        if isinstance(saved_epoch, int) and saved_epoch >= 0:
            start_epoch = saved_epoch + 1

    log(f"[Resume] checkpoint loaded from: {checkpoint_path}")
    log(f"[Resume] training will start from epoch index {start_epoch} (display epoch={start_epoch + 1}).")
    return start_epoch

def evaluate_metrics(
    model,
    dataloader,
    device,
    boundary_iou_ratio=0.02,
    boundary_f_ratio=0.008,
    nsd_ratio=0.008,
    prompt_mode="point",
    prediction_save_dir=None,
    feature_vis_prefix=None,
    epoch=None,
):
    arr = len(dataloader.dataset)
    numClass = 1
    if arr == 0:
        empty_metrics = {
            "mIoU": 0.0,
            "mDice": 0.0,
            "S_alpha": 0.0,
            "Fw_beta": 0.0,
            "E_phi": 0.0,
            "MAE": 0.0,
            "Boundary_IoU": 0.0,
            "Boundary_Fscore": 0.0,
            "Hausdorff": 0.0,
            "Hausdorff95": 0.0,
            "NSD": 0.0,
        }
        empty_metrics.update(_empty_fusion_stats())
        return empty_metrics

    area_intersection = np.zeros((numClass, arr), dtype=np.float64)
    area_union = np.zeros((numClass, arr), dtype=np.float64)
    area_sum = np.zeros((numClass, arr), dtype=np.float64)
    boundary_ious = []
    boundary_fs = []
    hausdorffs = []
    hausdorff95s = []
    nsds = []
    fusion_stat_values = {name: [] for name in FUSION_STAT_NAMES}
    sample_idx = 0

    FM, WFM, SM, EM, MAE, FMv2 = ff.init_metrics()

    for batch in dataloader:
        x = batch['image'].to(device)
        target = batch['label'].to(device)

        if prompt_mode == "point":
            point = batch.get('point', None)
            point_label = batch.get('point_label', None)
            if point is not None:
                point = point.to(device)
                prompt = (point, point_label.to(device)) if point_label is not None else point
            else:
                prompt = None
        elif prompt_mode == "bbox":
            bbox = batch.get('bbox', None)
            if bbox is not None:
                prompt = {"boxes": bbox.to(device)}
            else:
                prompt = None
        elif prompt_mode == "none":
            prompt = None
        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        batch_names = []
        for b in range(x.size(0)):
            sample_name = _get_eval_sample_name(dataloader, sample_idx, b)
            if feature_vis_prefix:
                sample_name = f"{feature_vis_prefix}_{sample_name}"
            batch_names.append(sample_name)

        pred, _, _, high_res_multimasks = model(x, prompt, image_names=batch_names, epoch=epoch)
        fusion_stats = getattr(model, "last_fusion_stats", {})
        for key in FUSION_STAT_NAMES:
            value = fusion_stats.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                fusion_stat_values[key].append(value)
        
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        pred_prob_tensor = torch.sigmoid(pred)
        pred_logit_np = pred.detach().float().cpu().numpy()
        pred_prob_save_np = pred_prob_tensor.detach().float().cpu().numpy()
        high_res_multimask_prob_np = None
        if prediction_save_dir is not None:
            if high_res_multimasks.shape[-2:] != target.shape[-2:]:
                high_res_multimasks = F.interpolate(
                    high_res_multimasks,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            high_res_multimask_prob_np = torch.sigmoid(high_res_multimasks).detach().float().cpu().numpy()
        pred_prob = pred_prob_tensor.detach().cpu().numpy()
        gt_np = target.detach().cpu().numpy()
        batch_size = pred_prob.shape[0]

        for b in range(batch_size):
            if prediction_save_dir is not None:
                sample_name = _get_eval_sample_name(dataloader, sample_idx, b)
                _save_prediction_outputs(
                    prediction_save_dir,
                    sample_name,
                    pred_logit_np[b, 0],
                    pred_prob_save_np[b, 0],
                    high_res_multimask_prob_np[b, 0],
                )

            pred_prob_b = pred_prob[b, 0]
            pred_mask = (pred_prob_b >= 0.5).astype(np.uint8)
            gt_mask = (gt_np[b, 0] >= 0.5).astype(np.uint8)

            (area_intersection[:, sample_idx], area_union[:, sample_idx], area_sum[:, sample_idx]) = intersectionAndUnion(
                pred_mask, gt_mask, numClass
            )

            pred_gray = (pred_prob_b * 255).astype(np.uint8)
            gt_gray = (gt_mask * 255).astype(np.uint8)

            FM.step(pred=pred_gray, gt=gt_gray)
            WFM.step(pred=pred_gray, gt=gt_gray)
            SM.step(pred=pred_gray, gt=gt_gray)
            EM.step(pred=pred_gray, gt=gt_gray)
            MAE.step(pred=pred_gray, gt=gt_gray)
            FMv2.step(pred=pred_gray, gt=gt_gray)

            boundary_ious.append(_boundary_iou(pred_mask, gt_mask, ratio=boundary_iou_ratio))
            boundary_fs.append(_boundary_f_score(pred_mask, gt_mask, ratio=boundary_f_ratio))
            hd, hd95 = _hd_hd95(pred_mask, gt_mask)
            hausdorffs.append(hd)
            hausdorff95s.append(hd95)
            nsds.append(_normalized_surface_dice(pred_mask, gt_mask, ratio=nsd_ratio))
            sample_idx += 1

    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)
    Dice = 1.0 * np.sum(2 * area_intersection, axis=1) / np.sum(np.spacing(1) + area_sum, axis=1)
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    metrics = {
        "mIoU": float(IoU.mean()),
        "mDice": float(Dice.mean()),
        "S_alpha": float(sm),
        "Fw_beta": float(wfm),
        "E_phi": float(em["curve"].mean()),
        "MAE": float(mae),
        "Boundary_IoU": _safe_mean(boundary_ious),
        "Boundary_Fscore": _safe_mean(boundary_fs),
        "Hausdorff": _safe_mean(hausdorffs),
        "Hausdorff95": _safe_mean(hausdorff95s),
        "NSD": _safe_mean(nsds),
    }
    for key in FUSION_STAT_NAMES:
        metrics[key] = _safe_mean(fusion_stat_values[key])
    return metrics

def evaluate_valid_sets(
    model,
    valid_dataloaders_point,
    valid_dataloaders_bbox,
    device,
    logger,
    args,
    prediction_root=None,
    epoch=None,
):
    if len(valid_dataloaders_point) == 0 and len(valid_dataloaders_bbox) == 0:
        logger.warning("[Internal Eval] No validation subsets are available.")
        return

    metric_names = [
        "mIoU",
        "mDice",
        "S_alpha",
        "Fw_beta",
        "E_phi",
        "MAE",
        "Boundary_IoU",
        "Boundary_Fscore",
        "Hausdorff",
        "Hausdorff95",
        "NSD",
    ]
    no_prompt_dataloaders = dict(valid_dataloaders_bbox)
    no_prompt_dataloaders.update(valid_dataloaders_point)
    eval_modes = [
        ("WITH POINT PROMPT", "point", valid_dataloaders_point),
        ("WITH BBOX PROMPT", "bbox", valid_dataloaders_bbox),
        ("WITHOUT PROMPT", "none", no_prompt_dataloaders),
    ]

    for mode_name, prompt_mode, mode_dataloaders in eval_modes:
        if len(mode_dataloaders) == 0:
            logger.warning(f"[Internal Eval] {mode_name} skipped: no compatible validation loader.")
            continue

        logger.info("=" * 96)
        logger.info(f"[Internal Eval] {mode_name}")
        logger.info("=" * 96)

        summary = {name: [] for name in metric_names}
        fusion_summary = {name: [] for name in FUSION_STAT_NAMES}

        for valid_name, valid_loader in mode_dataloaders.items():
            prediction_save_dir = None
            if prediction_root is not None:
                prediction_save_dir = os.path.join(
                    prediction_root,
                    _safe_path_name(prompt_mode),
                    _safe_path_name(valid_name),
                )
            metrics = evaluate_metrics(
                model=model,
                dataloader=valid_loader,
                device=device,
                boundary_iou_ratio=args.boundary_iou_ratio,
                boundary_f_ratio=args.boundary_f_ratio,
                nsd_ratio=args.nsd_ratio,
                prompt_mode=prompt_mode,
                prediction_save_dir=prediction_save_dir,
                feature_vis_prefix=f"{prompt_mode}_{valid_name}",
                epoch=epoch,
            )
            for key in metric_names:
                summary[key].append(metrics[key])
            for key in FUSION_STAT_NAMES:
                fusion_summary[key].append(metrics[key])
            logger.info(
                f"[Internal Eval] {valid_name}: "
                f"mDice={metrics['mDice']:.4f}, "
                f"mIoU={metrics['mIoU']:.4f}, "
                f"S_alpha={metrics['S_alpha']:.4f}, "
                f"Fw_beta={metrics['Fw_beta']:.4f}, "
                f"E_phi={metrics['E_phi']:.4f}, "
                f"MAE={metrics['MAE']:.4f}, "
                f"BoundaryIoU={metrics['Boundary_IoU']:.4f}, "
                f"BoundaryF={metrics['Boundary_Fscore']:.4f}, "
                f"HD={metrics['Hausdorff']:.4f}, "
                f"HD95={metrics['Hausdorff95']:.4f}, "
                f"NSD={metrics['NSD']:.4f}"
            )
            logger.info(f"[Fusion Eval] {valid_name}: {_format_fusion_stats(metrics)}")

        mean_metrics = {key: _safe_mean(summary[key]) for key in metric_names}
        mean_fusion_metrics = {key: _safe_mean(fusion_summary[key]) for key in FUSION_STAT_NAMES}

        logger.info(
            f"[Internal Eval] Mean({len(mode_dataloaders)} sets): "
            f"mDice={mean_metrics['mDice']:.4f}, "
            f"mIoU={mean_metrics['mIoU']:.4f}, "
            f"S_alpha={mean_metrics['S_alpha']:.4f}, "
            f"Fw_beta={mean_metrics['Fw_beta']:.4f}, "
            f"E_phi={mean_metrics['E_phi']:.4f}, "
            f"MAE={mean_metrics['MAE']:.4f}, "
            f"BoundaryIoU={mean_metrics['Boundary_IoU']:.4f}, "
            f"BoundaryF={mean_metrics['Boundary_Fscore']:.4f}, "
            f"HD={mean_metrics['Hausdorff']:.4f}, "
            f"HD95={mean_metrics['Hausdorff95']:.4f}, "
            f"NSD={mean_metrics['NSD']:.4f}"
        )
        logger.info(f"[Fusion Eval] Mean({len(mode_dataloaders)} sets): {_format_fusion_stats(mean_fusion_metrics)}")
        logger.info("-" * 96)


def main(args):    
    if args.train_prompt_type == "bbox":
        train_dataset = FullDataset_new_bbox(args.data_path, 352, mode='train')
        train_collate_fn = collate_fn_bbox
    else:
        train_dataset = FullDataset_new(args.data_path, 352, mode='train')
        train_collate_fn = collate_fn_multi_points

    # MFBFpnNeck 含 BatchNorm2d，batch_size=1 时训练报错，始终丢弃尾部不足一批的样本
        # 若最后一批只剩1张，BatchNorm2d会有问题
    if len(train_dataset) % args.batch_size == 1:
        drop_last = True
    else:
        drop_last = False
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=1, drop_last=drop_last,
                               collate_fn=train_collate_fn)

    device = torch.device("cuda")
    model = MMSAM2(
        args.hiera_path,
        feature_vis_enabled=args.save_feature_vis,
        feature_vis_dir=args.feature_vis_dir,
    )
    model.set_memory_mask_source(args.memory_mask_source, args.memory_mask_blend)
    model.to(device)

    MFBFpnNeck_params = (
                        []
                            + list(model.model.image_encoder.neck.mrb_convs.parameters())
                        )
    model_laste_params    =  (
                        []
                            + list(model.model.memory_attention.parameters())
                            + list(model.model.sam_mask_decoder.parameters())
                            + list(model.model.sam_prompt_encoder.parameters())
                        )
    
    model_params = list(
                            param for param in model.parameters() if id(param) not in {id(p) for p in MFBFpnNeck_params}
                        )
    
    model_params = list(
                            param for param in model_params if id(param) not in {id(p) for p in model_laste_params}
                       )


    optim = opt.AdamW([{"params":model_params,"initia_lr": args.lr},
                       {"params":MFBFpnNeck_params,"weight_decay": 1e-4},
                       {"params":model_laste_params,"weight_decay": 0},
                       ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    log_dir = ff.set_log_dir(os.path.join(args.save_path, args.exp_name))
    logger = ff.create_logger(log_dir['log'], phase='train')
  
    logger.info("-----------------------Training starts-----------------------")
    logger.info("Training with {} images".format(len(train_dataloader)))
    logger.info("Training prompt type: {}".format(args.train_prompt_type))
    logger.info("Feature visualization: {} ({})".format(args.save_feature_vis, args.feature_vis_dir))
    logger.info(
        "Memory mask source: {} (blend={})".format(
            args.memory_mask_source,
            args.memory_mask_blend,
        )
    )
    logger.info(
        "Loss weights: fuse=1.0, sam_aux={}, unet_side={}".format(
            args.sam_aux_loss_weight,
            args.unet_side_loss_weight,
        )
    )
    logger.info("args:{}".format(args))

    start_epoch = 0
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        start_epoch = load_training_checkpoint(
            model=model,
            checkpoint_path=args.resume_checkpoint,
            device=device,
            optimizer=optim,
            scheduler=scheduler,
            logger=logger,
            resume_memory_bank=args.resume_memory_bank,
        )
        if start_epoch >= args.epoch:
            logger.warning(
                f"[Resume] start_epoch={start_epoch} is already >= total epoch={args.epoch}. "
                "Nothing to train. Increase --epoch to continue."
            )
            return

    model.set_etis_boundary_optimized_fusion(
        enabled=args.etis_boundary_optimized_fusion,
        opposite_scale=args.etis_opposite_residual_scale,
        confidence_margin=args.etis_confidence_margin,
    )
    logger.info(
        "ETIS boundary optimized fusion: {} (opposite_scale={}, confidence_margin={})".format(
            args.etis_boundary_optimized_fusion,
            args.etis_opposite_residual_scale,
            args.etis_confidence_margin,
        )
    )

    valid_dataloaders_point = {}
    valid_dataloaders_bbox = {}
    for valid_name in args.valid_list:
        try:
            valid_dataset_point = FullDataset_new(args.data_path, 352, mode='valid', valid_file=valid_name)
            valid_dataloaders_point[valid_name] = DataLoader(
                valid_dataset_point,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                collate_fn=collate_fn_multi_points,
            )
            logger.info(f"[Internal Eval] Loaded point valid subset {valid_name}: {len(valid_dataset_point)} images")
        except Exception as e:
            logger.warning(f"[Internal Eval] Failed to load point valid subset {valid_name}: {e}")

        try:
            valid_dataset_bbox = FullDataset_new_bbox(args.data_path, 352, mode='valid', valid_file=valid_name)
            valid_dataloaders_bbox[valid_name] = DataLoader(
                valid_dataset_bbox,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                collate_fn=collate_fn_bbox,
            )
            logger.info(f"[Internal Eval] Loaded bbox valid subset {valid_name}: {len(valid_dataset_bbox)} images")
        except Exception as e:
            logger.warning(f"[Internal Eval] Failed to load bbox valid subset {valid_name}: {e}")

    for epoch in range(start_epoch, args.epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            
            # Prompt Dropout strategy: 50% probability to drop prompts
            if random.random() < 0.5:
                current_click = None
            else:
                if args.train_prompt_type == "bbox":
                    bbox = batch['bbox'].to(device)
                    current_click = {"boxes": bbox}
                else:
                    point = batch['point'].to(device)
                    point_label = batch.get('point_label', None)
                    if point_label is not None:
                        point_label = point_label.to(device)
                    current_click = (point, point_label) if point_label is not None else point
                
            optim.zero_grad()
            pred0, pred1, pred2, sam_pred = model(x, current_click)
            loss_fuse = structure_loss(pred0, target)
            loss_sam = structure_loss(sam_pred, target)
            loss_unet1 = structure_loss(pred1, target)
            loss_unet2 = structure_loss(pred2, target)
            loss = (
                loss_fuse
                + args.sam_aux_loss_weight * loss_sam
                + args.unet_side_loss_weight * (loss_unet1 + loss_unet2)
            )
            loss.backward()
            optim.step()
            if i % 50 == 0:
                logger.info(
                    "epoch:{}-{}: loss:{:.6f}, fuse:{:.6f}, sam:{:.6f}, unet1:{:.6f}, unet2:{:.6f}".format(
                        epoch + 1,
                        i + 1,
                        loss.item(),
                        loss_fuse.item(),
                        loss_sam.item(),
                        loss_unet1.item(),
                        loss_unet2.item(),
                    )
                )
        scheduler.step()

        if (epoch + 1) % args.valid_interval != 0 and (epoch + 1) != args.epoch:
            continue

        model.eval()
        with torch.no_grad():
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
            ckpt_name = f'{args.exp_name.lower()}_{epoch + 1}_{timestamp}.pth'
            ckpt_path = os.path.join(log_dir['checkpoints'], ckpt_name)
            
            memory_bank_state = _memory_bank_state(model)
            if len(memory_bank_state['memories']) != len(model.memory_bank.memories):
                logger.warning(
                    "[Checkpoint] skipped invalid DMB memories while saving: {} / {}".format(
                        len(model.memory_bank.memories) - len(memory_bank_state['memories']),
                        len(model.memory_bank.memories),
                    )
                )

            # Save memory bank state along with model state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'memory_bank_state': memory_bank_state,
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, ckpt_path)
            
            logger.info(f'Checkpoint saved: {ckpt_path}')
            prediction_root = None
            if args.save_predictions:
                prediction_root = os.path.join(
                    os.path.dirname(log_dir['checkpoints']),
                    "predictions",
                    f"epoch_{epoch + 1:03d}_{timestamp}",
                )

            evaluate_valid_sets(
                model,
                valid_dataloaders_point,
                valid_dataloaders_bbox,
                device,
                logger,
                args,
                prediction_root=prediction_root,
                epoch=epoch + 1,
            )
            if prediction_root is not None:
                logger.info(f'[Internal Eval] Prediction outputs saved under: {prediction_root}')

# 1024, 2024, 3407
def seed_torch(seed=1024):

	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    if args.seed is not None:
        seed_torch(args.seed)
    main(args)
