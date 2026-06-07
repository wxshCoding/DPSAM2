import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
# from SAM2UNet import SAM2UNet
from mmsam2 import MMSAM2
from dataset import TestDataset
import shutil


parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", type=str, default='./7.pth', help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, default='./data/Marine/valid/RMAS/images/', help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, default='./data/Marine/valid/RMAS/masks/', help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, default='./data/Marine/valid/RMAS/preds/', help="path to save the predicted masks")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = MMSAM2().to(device)


checkpoint = torch.load(args.checkpoint, map_location=device)

if 'model_state_dict' in checkpoint:
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)


if 'memory_bank_state' in checkpoint:
    memory_state = checkpoint['memory_bank_state']
    
    # 适配张量所在的设备
    memories = memory_state.get('memories', [])
    if memories:
        device_memories = []
        for memory in memories:
            device_memory = []
            for item in memory:
                if isinstance(item, torch.Tensor):
                    device_memory.append(item.to(device))
                else:
                    device_memory.append(item)
            device_memories.append(device_memory)
        model.memory_bank.memories = device_memories
    else:
        model.memory_bank.memories = []

    model.memory_bank.max_size = memory_state.get('max_size', 12)
    model.memory_bank.min_size = memory_state.get('min_size', 6)
    model.memory_bank.similarity_threshold = memory_state.get('similarity_threshold', 0.85)
    model.memory_bank.decay_factor = memory_state.get('decay_factor', 0.98)
    model.memory_bank.usage_counts = memory_state.get('usage_counts', [])
    model.memory_bank.timestamps = memory_state.get('timestamps', [])
    model.memory_bank.current_time = memory_state.get('current_time', 0)
    print("Memory bank state loaded successfully")
else:
    print("No memory bank state found in checkpoint. Memory bank will start empty.")

model.eval()
model.cuda()
if os.path.exists(args.save_path):
    shutil.rmtree(args.save_path)
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name,click = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        
        # True Prompt-Free inference: pass click=None
        res, _, _ = model(image, None)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        # print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
