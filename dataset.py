from scipy import ndimage
import torchvision.transforms.functional as F
import numpy as np
import random
import os
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


def collate_fn_multi_points(batch):
    """
    Pad variable-length point prompts to batch tensors.
    Returns:
      - point:       [B, Pmax, 2] float32
      - point_label: [B, Pmax] int64, padded with -1
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    point_list = []
    point_label_list = []
    for item in batch:
        points = np.asarray(item["point"])
        labels_np = np.asarray(item["point_label"])
        if points.ndim == 1:
            points = points[None, :]
        if labels_np.ndim == 0:
            labels_np = labels_np[None]
        point_list.append(points.astype(np.float32))
        point_label_list.append(labels_np.astype(np.int64))

    max_points = max(points.shape[0] for points in point_list)
    batch_size = len(batch)
    point_tensor = torch.zeros((batch_size, max_points, 2), dtype=torch.float32)
    point_label_tensor = torch.full((batch_size, max_points), -1, dtype=torch.int64)

    for i, (points, labels_np) in enumerate(zip(point_list, point_label_list)):
        n = points.shape[0]
        point_tensor[i, :n] = torch.from_numpy(points)
        point_label_tensor[i, :n] = torch.from_numpy(labels_np)

    return {
        "image": images,
        "label": labels,
        "point": point_tensor,
        "point_label": point_label_tensor,
    }


def collate_fn_bbox(batch):
    """
    Collate batch for bbox prompts.
    Returns:
      - bbox: [B, 4] float32 in (x_min, y_min, x_max, y_max)
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    bboxes = []
    for item in batch:
        bbox = np.asarray(item["bbox"], dtype=np.float32).reshape(-1)
        if bbox.shape[0] != 4:
            raise ValueError(f"Unsupported bbox shape: {bbox.shape}")
        bboxes.append(bbox)

    bbox_tensor = torch.from_numpy(np.stack(bboxes, axis=0)).to(dtype=torch.float32)
    return {
        "image": images,
        "label": labels,
        "bbox": bbox_tensor,
    }



class FullDataset_new(Dataset):
    def __init__(self, data_root, size, mode = 'train',valid_file = ''):
        super(FullDataset_new, self).__init__()
        if mode == 'train':
            self.image_root = os.path.join(data_root, 'train/images')
            self.gt_root = os.path.join(data_root, 'train/masks')
        else:
            self.image_root = os.path.join(data_root,'valid',valid_file, 'images')
            self.gt_root = os.path.join(data_root, 'valid',valid_file,'masks')

        self.images = [os.path.join(self.image_root,f) for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(self.gt_root,f) for f in os.listdir(self.gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                    Resize((size, size)),
                    ToTensor(),
                    Normalize()
            ])
            
    def __getitem__(self, idx):
            image = self.rgb_loader(self.images[idx])
            label = self.binary_loader(self.gts[idx])
            data = {'image': image, 'label': label}
            data = self.transform(data)
            mask_click  = data["label"].clone()
            point_label, pt_cup = self.random_click_per_region(np.array(mask_click).squeeze(0), point_label = 1)
            data["point"] = pt_cup
            data["point_label"] = point_label
            return data
    
    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def random_click_per_region(self, mask, point_label=1):
        """
        每个白色连通区域随机采样一个点。
        返回:
        - point_labels: [N]，每个点对应标签(前景=1; 全背景时返回[0])
        - points: [N, 2]，每个点坐标为[y, x]

        采样策略:
        - 前景存在时: 每个连通域采样1个点
        - 前景不存在时: 返回1个背景点, point_label=0
        """
        mask = np.asarray(mask)
        fg_mask = mask > 0.5

        # 全背景样本: 返回一个背景点
        if not np.any(fg_mask):
            bg_indices = np.argwhere(~fg_mask)
            if len(bg_indices) == 0:
                all_indices = np.argwhere(np.ones_like(mask, dtype=bool))
                point = all_indices[np.random.randint(len(all_indices))]
                return np.array([0], dtype=np.int64), np.array([point], dtype=np.int64)
            point = bg_indices[np.random.randint(len(bg_indices))]
            return np.array([0], dtype=np.int64), np.array([point], dtype=np.int64)

        structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
        labeled_array, num_features = ndimage.label(fg_mask, structure=structure)
        if num_features <= 0:
            fg_indices = np.argwhere(fg_mask)
            point = fg_indices[np.random.randint(len(fg_indices))]
            return np.array([point_label], dtype=np.int64), np.array([point], dtype=np.int64)

        all_points = []
        for region_id in range(1, num_features + 1):
            region_indices = np.argwhere(labeled_array == region_id)
            if len(region_indices) == 0:
                continue
            point = region_indices[np.random.randint(len(region_indices))]
            all_points.append(point)

        if len(all_points) == 0:
            fg_indices = np.argwhere(fg_mask)
            point = fg_indices[np.random.randint(len(fg_indices))]
            return np.array([point_label], dtype=np.int64), np.array([point], dtype=np.int64)

        points = np.array(all_points, dtype=np.int64)
        labels = np.full((points.shape[0],), point_label, dtype=np.int64)
        return labels, points
      
    def random_click(self , mask, point_label = 1):
        return self.random_click_per_region(mask, point_label)
    
          

class FullDataset_new_bbox(FullDataset_new):
    """
    Keep FullDataset_new as point-prompt mode, and provide bbox-prompt mode
    for training.
    """
    def __getitem__(self, idx):
            image = self.rgb_loader(self.images[idx])
            label = self.binary_loader(self.gts[idx])
            data = {'image': image, 'label': label}
            data = self.transform(data)
            mask_click = np.asarray(data["label"]).squeeze(0)
            data["bbox"] = self.mask_to_bbox(mask_click)
            return data

    @staticmethod
    def mask_to_bbox(mask):
        mask = np.asarray(mask)
        fg_mask = mask > 0.5
        if not np.any(fg_mask):
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        ys, xs = np.where(fg_mask)
        x_min = float(np.min(xs))
        x_max = float(np.max(xs))
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root,f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts =    [os.path.join(gt_root   ,f) for f in os.listdir(gt_root   ) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[idx])
        gt = np.array(gt)

        # name = self.images[idx].split('/')[-1]

        data = {'image': image, 'label': gt}
        return data
    
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]
        mask_click  = gt.copy()
        point_label, pt_cup = self.random_click_per_region(np.array(mask_click), point_label = 1)
        self.index += 1
        return image, gt, name, pt_cup, point_label
    

    def random_click(self , mask, point_label = 1):
        return self.random_click_per_region(mask, point_label)

    def random_click_per_region(self, mask, point_label=1):
        """
        每个白色连通区域随机采样一个点。
        返回:
        - point_labels: [N]
        - points: [N, 2]
        """
        mask = np.asarray(mask)
        fg_mask = mask > 0.5

        if not np.any(fg_mask):
            bg_indices = np.argwhere(~fg_mask)
            if len(bg_indices) == 0:
                all_indices = np.argwhere(np.ones_like(mask, dtype=bool))
                point = all_indices[np.random.randint(len(all_indices))]
                return np.array([0], dtype=np.int64), np.array([point], dtype=np.int64)
            point = bg_indices[np.random.randint(len(bg_indices))]
            return np.array([0], dtype=np.int64), np.array([point], dtype=np.int64)

        structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
        labeled_array, num_features = ndimage.label(fg_mask, structure=structure)
        if num_features <= 0:
            fg_indices = np.argwhere(fg_mask)
            point = fg_indices[np.random.randint(len(fg_indices))]
            return np.array([point_label], dtype=np.int64), np.array([point], dtype=np.int64)

        all_points = []
        for region_id in range(1, num_features + 1):
            region_indices = np.argwhere(labeled_array == region_id)
            if len(region_indices) == 0:
                continue
            point = region_indices[np.random.randint(len(region_indices))]
            all_points.append(point)

        if len(all_points) == 0:
            fg_indices = np.argwhere(fg_mask)
            point = fg_indices[np.random.randint(len(fg_indices))]
            return np.array([point_label], dtype=np.int64), np.array([point], dtype=np.int64)

        points = np.array(all_points, dtype=np.int64)
        labels = np.full((points.shape[0],), point_label, dtype=np.int64)
        return labels, points

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
