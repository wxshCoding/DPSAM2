import os
import torch
import torch.nn as nn
# import numpy as np
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import math
from sam2.modeling.backbones.MFB import MFB_modified
from _utils import plot_feature_map
# from sam2.sam2_image_predictor import SAM2ImagePredictor


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""
    # todo 可以将in_channels 参数删除，因为是可以推断的关系
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 进行创新
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

        # self.down_proj = nn.Linear(dim, 32)
        # self.act = nn.GELU()
        # self.up_proj = nn.Linear(32, dim)

        # nn.init.zeros_(self.up_proj.weight)
        # nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        prompt = self.prompt_learn(x)
        # prompt = self.up_proj(self.act(self.down_proj(x)))
        promped = x + prompt
        net = self.block(promped)
        return net

# 1、提取bak_function.py中的函数
# 2、使用 memory_attention sam_prompt_encoder sam_mask_decoder _encode_new_memory 等函数
# 3、将net 作为参数传入memory_forward 初始化中
# 4、bak_function.train_sam中代码逻辑封装进入memory_forward 中

class DynamicMemoryBank():
    def __init__(self, max_size=12, min_size=4, similarity_threshold=0.85, decay_factor=0.98):
        self.memories = []
        self.max_size = max_size
        self.min_size = min_size
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor
        self.usage_counts = []
        self.timestamps = []
        self.current_time = 0

        
    def update(self, new_feature, pos_enc, iou_score, image_embed, current_mm=None):
        """
        Update memory bank with new feature, using similarity and IoU-based strategies.
        
        Args:
            new_feature: The new feature to possibly add to memory
            pos_enc: Position encoding for the feature
            iou_score: IoU score of the feature's mask prediction
            image_embed: Global image embedding for similarity comparison
            current_mm: Optional multi-scale context feature for blending
        """
        self.current_time += 1
        
        # If empty or not enough memories yet, just add
        if len(self.memories) < self.min_size:
            self._add_memory(new_feature, pos_enc, iou_score, image_embed)
            return
            
        # Compute similarities with existing memories
        memory_features = [m[0].reshape(-1) for m in self.memories]
        memory_features = torch.stack(memory_features)
        new_feature_flat = new_feature.reshape(-1)
        
        # Normalize for cosine simi larity
        memory_norm = F.normalize(memory_features, p=2, dim=1)
        new_feature_flat = new_feature_flat.to(dtype=memory_norm.dtype)
        new_norm = F.normalize(new_feature_flat, p=2, dim=0).unsqueeze(1)
        
        # Calculate similarity scores between the new feature and all memories
        similarity_scores = torch.mm(memory_norm, new_norm).squeeze() # 新特征与所有记忆的相似度
        
        # Calculate similarity matrix between all existing memories
        similarity_matrix = torch.mm(memory_norm, memory_norm.t()) # 所有记忆之间的相似度
        
        # Create version with diagonal set to -inf for finding max similarities
        similarity_matrix_no_diag = similarity_matrix.clone()
        diag_indices = torch.arange(similarity_matrix_no_diag.size(0))
        similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
        
        # STRATEGY 1: If very similar to existing memory, merge instead of add
        max_sim, max_idx = torch.max(similarity_scores, dim=0)
        if max_sim > self.similarity_threshold:
            # Update existing memory with weighted average
            alpha = self.adaptive_blend_factor(iou_score, self.memories[max_idx][2], 
                                        self.usage_counts[max_idx])
            
            # If multi-scale context is provided, use blended feature
            if current_mm is not None:
                blend_factor = 0.5  # Could be made a parameter or adaptive
                push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
            else:
                push_feature = new_feature
                
            self.memories[max_idx][0] = alpha * push_feature + (1 - alpha) * self.memories[max_idx][0]
            self.memories[max_idx][2] = max(iou_score, self.memories[max_idx][2])  # Take best IoU
            self.usage_counts[max_idx] += 1
            self.timestamps[max_idx] = self.current_time
            return
                
        # STRATEGY 2: Find the memory with minimum similarity to the new feature
        min_similarity_index = torch.argmin(similarity_scores)
        
        # Find the memory that has highest similarity to this minimum-similar memory
        max_similarity_index = torch.argmax(similarity_matrix_no_diag[min_similarity_index])
        
        # If new feature is less similar to min_similarity_index than max_similarity_index
        # This means we're replacing a memory that already has a similar representation
        if similarity_scores[min_similarity_index] < similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
            # Only if new IoU is better (with tolerance)
            if iou_score > self.memories[max_similarity_index][2] - 0.1:
                # Create a blended feature if multi-scale context available
                if current_mm is not None:
                    blend_factor = 0.5  # Could be self.scale_factor
                    push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
                else:
                    push_feature = new_feature
                    
                # Remove the redundant memory
                self.memories.pop(max_similarity_index)
                self.usage_counts.pop(max_similarity_index)
                self.timestamps.pop(max_similarity_index)
                
                # Add the new memory
                self._add_memory(push_feature, pos_enc, iou_score, image_embed)
                return
        
        # STRATEGY 3: If needs to add but no clear replacement based on similarity
        if len(self.memories) >= self.max_size:
            # Calculate removal score based on uniqueness, usage and age
            removal_scores = self._calculate_removal_scores(similarity_scores)
            remove_idx = torch.argmax(removal_scores).item()
            
            # Remove the memory with highest removal score
            self.memories.pop(remove_idx)
            self.usage_counts.pop(remove_idx)
            self.timestamps.pop(remove_idx)
        
        # Blend with multi-scale context if available
        if current_mm is not None:
            blend_factor = 0.5  # Could be self.scale_factor
            push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
        else:
            push_feature = new_feature
            
        # Add the new memory
        self._add_memory(push_feature, pos_enc, iou_score, image_embed)  
    
    #  暂时不用
    def update_similarity(self, new_feature, pos_enc, iou_score, image_embed):
        self.current_time += 1
        
        # If empty, just add
        if len(self.memories) < self.min_size:
            self._add_memory(new_feature, pos_enc, iou_score, image_embed)
            return
            
        # Compute similarities with existing memories
        memory_features = [m[0].reshape(-1) for m in self.memories]
        memory_features = torch.stack(memory_features)
        new_feature_flat = new_feature.reshape(-1)
        
        # Normalize for cosine similarity
        memory_norm = F.normalize(memory_features, p=2, dim=1)
            # Ensure consistent data type before normalization
        new_feature_flat = new_feature_flat.to(dtype=memory_norm.dtype)
        new_norm = F.normalize(new_feature_flat, p=2, dim=0).unsqueeze(1)
        
        # Calculate similarity scores
        similarities = torch.mm(memory_norm, new_norm).squeeze()
        
        # If very similar to existing memory, merge instead of add
        max_sim, max_idx = torch.max(similarities, dim=0)
        if max_sim > self.similarity_threshold:
            # Update existing memory with weighted average
            alpha = self.adaptive_blend_factor(iou_score, self.memories[max_idx][2], 
                                           self.usage_counts[max_idx])
            self.memories[max_idx][0] = alpha * new_feature + (1 - alpha) * self.memories[max_idx][0]
            self.memories[max_idx][2] = max(iou_score, self.memories[max_idx][2])  # Take best IoU
            self.usage_counts[max_idx] += 1
            self.timestamps[max_idx] = self.current_time
            return
            
        # Need to add new memory - check if we need to remove one
        if len(self.memories) >= self.max_size:
            # Calculate removal score based on:
            # 1. Low similarity to other memories (unique)
            # 2. Low usage count (not frequently needed)
            # 3. Old (not recently accessed)
            removal_scores = self._calculate_removal_scores(similarities)
            remove_idx = torch.argmax(removal_scores).item()
            
            # Remove the memory with highest removal score
            self.memories.pop(remove_idx)
            self.usage_counts.pop(remove_idx)
            self.timestamps.pop(remove_idx)
            
        # Add the new memory
        self._add_memory(new_feature, pos_enc, iou_score, image_embed)
        
    def _add_memory(self, feature, pos_enc, iou, embed):
        self.memories.append([feature.detach(), pos_enc.detach(), iou, embed.detach()])
        self.usage_counts.append(1)
        self.timestamps.append(self.current_time)
        
    def _calculate_removal_scores(self, current_similarities):

        memory_matrix = torch.stack([m[0].reshape(-1) for m in self.memories])
        memory_norm = F.normalize(memory_matrix, p=2, dim=1)
        
        # Calculate similarity matrix between all memories
        sim_matrix = torch.mm(memory_norm, memory_norm.t())
        
        # Device to use for all tensors
        device = sim_matrix.device
        
        # Calculate uniqueness (low similarity to other memories is good to keep)
        sim_matrix.fill_diagonal_(0)  # Remove self-similarity
        # uniqueness = 1 - torch.mean(sim_matrix, dim=1)
        uniqueness = torch.mean(sim_matrix, dim=1)
        
        # Age factor (newer is better to keep)
        max_time = float(self.current_time)
        age_factor = torch.tensor([(max_time - t) / max_time for t in self.timestamps], device=device)
        
        # Usage factor (more used is better to keep)
        usage = torch.tensor([1.0 / (c + 1) for c in self.usage_counts], device=device)
        
        # Combine factors - higher score means more likely to remove
        # removal_scores = 0.4 * uniqueness + 0.3 * age_factor + 0.3 * usage
        removal_scores = 0.6 * uniqueness + 0.2 * age_factor + 0.2 * usage
        return removal_scores

        
    def adaptive_blend_factor(self, new_iou, old_iou, usage_count):
        # # More useful (higher IoU) and less used memories get updated more
        # iou_factor = torch.sigmoid(torch.tensor(new_iou - old_iou + 0.1)) * 0.5 + 0.25
        # usage_factor = 1.0 / (1 + math.log(1 + usage_count))
        # return float(iou_factor * usage_factor + 0.3)  # Ensure some minimum update
            # More useful (higher IoU) and less used memories get updated more
        # diff = 0.1
        if isinstance(new_iou, torch.Tensor) or isinstance(old_iou, torch.Tensor):
            # Handle tensor inputs properly
            if isinstance(new_iou, torch.Tensor):
                new_iou_val = new_iou.item()
            else:
                new_iou_val = new_iou
                
            if isinstance(old_iou, torch.Tensor):
                old_iou_val = old_iou.item()
            else:
                old_iou_val = old_iou
                
            diff_val = new_iou_val - old_iou_val + 0.1
            # Create tensor on the same device as input tensors if they're tensors
            device = new_iou.device if isinstance(new_iou, torch.Tensor) else (
                old_iou.device if isinstance(old_iou, torch.Tensor) else None)
            diff_tensor = torch.tensor(diff_val, device=device)
        else:
            # Handle scalar inputs
            diff_tensor = torch.tensor(new_iou - old_iou + 0.1)
        
        iou_factor = torch.sigmoid(diff_tensor) * 0.5 + 0.25
        usage_factor = 1.0 / (1 + math.log(1 + usage_count))
        return float(iou_factor * usage_factor + 0.3)  # Ensure some minimum update
    
    # 提取最相似的记忆   
    def retrieve(self, query_embedding, top_k=None):
        if not self.memories:
            return None, None

        # Normalize query
        query_embedding = torch.nan_to_num(query_embedding.float(), nan=0.0, posinf=0.0, neginf=0.0)
        query_norm = F.normalize(query_embedding, p=2, dim=0, eps=1e-6)
        
        # Get all memory embeddings 
        memory_embeds = torch.stack([
            torch.nan_to_num(m[3].float(), nan=0.0, posinf=0.0, neginf=0.0)
            for m in self.memories
        ])
        memory_norm = F.normalize(memory_embeds, p=2, dim=1, eps=1e-6)
        
        # Calculate similarity
        similarities = torch.mm(memory_norm, query_norm.unsqueeze(1)).squeeze()
        similarities = torch.nan_to_num(similarities, nan=-1.0, posinf=1.0, neginf=-1.0)
        

           # Handle case when there's only one memory (similarities becomes 0-d tensor)
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
        # Get most similar memories
        if top_k is None:
            top_k = len(self.memories)
        
        _, indices = torch.topk(similarities, min(top_k, len(self.memories)))


            # Handle case when indices is 0-d tensor (single memory)
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        # Update usage counts for retrieved memories
        index_list = [int(idx.item()) for idx in indices]
        for idx in index_list:
            self.usage_counts[idx] += 1
            self.timestamps[idx] = self.current_time
            
        return [self.memories[i] for i in index_list], similarities[indices]
    
class MMSAM2(nn.Module):
    def __init__(self, checkpoint_path=None, feature_vis_enabled=False, feature_vis_dir="feature_vis") -> None:
        super(MMSAM2, self).__init__()
        # SAM2 推理开启了 bfloat16 + TF32，GPU 上会有非确定性的结果，训练时也会有一定程度的非确定性（尤其是小批量），但可以通过设置随机种子和一些环境变量来尽量减少这种非确定性。
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            self.model = build_sam2(model_cfg, checkpoint_path)
        else:
            self.model = build_sam2(model_cfg)

        del self.model.mask_downsample
        del self.model.obj_ptr_tpos_proj
        del self.model.obj_ptr_proj
      
        # self.image_encoder = self.model.image_encoder

        # 动态记忆库
        self.memory_bank = DynamicMemoryBank(max_size=20, min_size=6,similarity_threshold=0.85, decay_factor=0.98)
        self.down_2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.scale_factor = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 添加多尺度特征融合的卷积层
        # 注意：这里使用固定的通道数，基于 SAM2 的默认配置
        hidden_dim = 256  # 从 model.hidden_dim 获取，通常是 256
        self.conv_s2 = nn.Conv2d(64, hidden_dim, kernel_size=1, stride=1)  # 中间尺度
        self.conv_s3 = nn.Conv2d(32, hidden_dim, kernel_size=1, stride=1)  # 大尺度
        
        # MFB 特征增强模块
        # 基于 SAM2 memory encoder 的输出通道数，通常是 64
        memory_dim = 64
        self.feat_small = MFB_modified(memory_dim, memory_dim)
        # self.feat_mid = MFB_modified(memory_dim, memory_dim)  
        # self.feat_large = MFB_modified(memory_dim, memory_dim)


        # - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # freze the encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.image_encoder.neck.mrb_convs.parameters():
            param.requires_grad = True

        blocks = []
        for block in self.model.image_encoder.trunk.blocks:
            blocks.append(
                Adapter(block)
            )
        self.model.image_encoder.trunk.blocks = nn.Sequential(
            *blocks
        )


        # for param in self.model.image_encoder.trunk.blocks.parameters():
        #     param.requires_grad = True
        self.up1 = (Up(512, 256))#  与 yaml 中 d_model 相对应 128 = 64*2 
        self.up2 = (Up(512, 256))
        self.up3 = (Up(512, 256))
        self.up4 = (Up(512, 256))

        #todo 优化 128=>1 很突兀，建议降的速度慢一些
        self.side1 = nn.Conv2d(256, 1, kernel_size=1)
        self.side2 = nn.Conv2d(256, 1, kernel_size=1)
        self.head = nn.Conv2d(256, 1, kernel_size=1)

        # Kept for checkpoint compatibility; the current residual fusion does not use T1/T2.
        self.T1 = nn.Parameter(torch.ones(1))
        self.T2 = nn.Parameter(torch.ones(1))
        # SAM-dominant residual fusion controls:
        # - max_detail_gain is the ceiling for the U-Net-style residual strength.
        # - detail_gate_floor keeps a minimum residual path even when SAM2 is confident,
        #   so high-confidence SAM2 mistakes can still be weakly corrected.
        # - detail_gain_logit is the learnable strength knob; sigmoid keeps the actual
        #   gain in (0, max_detail_gain). The initial value gives gain ~= 0.5.
        self.max_detail_gain = 2.0
        self.detail_gate_floor = 0.15
        self.detail_gate_temperature = 6.0
        self.detail_boundary_kernel = 7
        self.detail_gain_logit = nn.Parameter(torch.tensor(-1.0986123))
        self.register_buffer("etis_boundary_optimized_fusion_flag", torch.tensor(0.0), persistent=True)
        self.register_buffer("etis_opposite_residual_scale", torch.tensor(0.35), persistent=True)
        self.register_buffer("etis_confidence_margin", torch.tensor(4.0), persistent=True)
        self.feature_vis_enabled = feature_vis_enabled
        self.feature_vis_dir = feature_vis_dir
        self.memory_mask_source = "fused"  # "sam", "fused", or "blend"
        self.memory_mask_blend = 0.7
        self.last_fusion_stats = {}

    def set_feature_visualization(self, enabled=True, save_dir=None):
        self.feature_vis_enabled = bool(enabled)
        if save_dir is not None:
            self.feature_vis_dir = save_dir

    def set_memory_mask_source(self, source="fused", blend=0.7):
        if source not in {"sam", "fused", "blend"}:
            raise ValueError("memory mask source must be one of: sam, fused, blend")
        self.memory_mask_source = source
        self.memory_mask_blend = blend

    def set_etis_boundary_optimized_fusion(self, enabled=False, opposite_scale=0.35, confidence_margin=4.0):
        self.etis_boundary_optimized_fusion_flag.fill_(1.0 if enabled else 0.0)
        self.etis_opposite_residual_scale.fill_(float(opposite_scale))
        self.etis_confidence_margin.fill_(float(confidence_margin))

    def _record_fusion_stats(
        self,
        sam_term,
        unet_term,
        fused_term,
        detail_gain=None,
        detail_gate=None,
        boundary_gate=None,
        opposite_scale=None,
        opposite_ratio=None,
    ):
        with torch.no_grad():
            sam = torch.nan_to_num(sam_term.detach().float(), nan=0.0, posinf=50.0, neginf=-50.0)
            unet = torch.nan_to_num(unet_term.detach().float(), nan=0.0, posinf=50.0, neginf=-50.0)
            fused = torch.nan_to_num(fused_term.detach().float(), nan=0.0, posinf=50.0, neginf=-50.0)

            sam_abs = sam.abs().mean()
            unet_abs = unet.abs().mean()
            fused_abs = fused.abs().mean()

            sam_sign = sam.sign()
            unet_sign = unet.sign()
            fused_sign = fused.sign()

            same_sign = (sam_sign == unet_sign).float().mean()
            conflict = ((sam_sign * unet_sign) < 0).float().mean()
            fused_sam_agree = (fused_sign == sam_sign).float().mean()
            fused_unet_agree = (fused_sign == unet_sign).float().mean()

            sam_mask = torch.sigmoid(sam) >= 0.5
            unet_mask = torch.sigmoid(unet) >= 0.5
            fused_mask = torch.sigmoid(fused) >= 0.5

            eps = torch.tensor(1e-6, device=sam.device)
            sam_fg = sam_mask.float().mean()
            unet_fg = unet_mask.float().mean()
            fused_fg = fused_mask.float().mean()
            sam_keep = (sam_mask & fused_mask).float().sum() / (sam_mask.float().sum() + eps)
            unet_keep = (unet_mask & fused_mask).float().sum() / (unet_mask.float().sum() + eps)
            sam_removed = (sam_mask & ~fused_mask).float().sum() / (sam_mask.float().sum() + eps)

            self.last_fusion_stats = {
                "fusion_T1": float(self.T1.detach().float().cpu().item()),
                "fusion_T2": float(self.T2.detach().float().cpu().item()),
                "fusion_detail_gain": float(detail_gain.detach().float().cpu().item()) if detail_gain is not None else 0.0,
                "fusion_detail_gate_mean": float(detail_gate.detach().float().mean().cpu().item()) if detail_gate is not None else 0.0,
                "fusion_boundary_gate_mean": float(boundary_gate.detach().float().mean().cpu().item()) if boundary_gate is not None else 0.0,
                "fusion_etis_opt": float(self.etis_boundary_optimized_fusion_flag.detach().float().cpu().item()),
                "fusion_opposite_scale_mean": float(opposite_scale.detach().float().mean().cpu().item()) if opposite_scale is not None else 1.0,
                "fusion_opposite_ratio": float(opposite_ratio.detach().float().cpu().item()) if opposite_ratio is not None else 0.0,
                "fusion_sam_abs_mean": float(sam_abs.cpu().item()),
                "fusion_unet_abs_mean": float(unet_abs.cpu().item()),
                "fusion_fused_abs_mean": float(fused_abs.cpu().item()),
                "fusion_unet_to_sam_abs": float((unet_abs / (sam_abs + eps)).cpu().item()),
                "fusion_same_sign": float(same_sign.cpu().item()),
                "fusion_conflict": float(conflict.cpu().item()),
                "fusion_pr_sam_agree": float(fused_sam_agree.cpu().item()),
                "fusion_pr_unet_agree": float(fused_unet_agree.cpu().item()),
                "fusion_sam_fg": float(sam_fg.cpu().item()),
                "fusion_unet_fg": float(unet_fg.cpu().item()),
                "fusion_pr_fg": float(fused_fg.cpu().item()),
                "fusion_sam_keep": float(sam_keep.cpu().item()),
                "fusion_unet_keep": float(unet_keep.cpu().item()),
                "fusion_sam_removed": float(sam_removed.cpu().item()),
            }

    @staticmethod
    def _safe_probabilities(scores, dim):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        probs = F.softmax(scores, dim=dim)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        sums = probs.sum(dim=dim, keepdim=True)
        valid = sums > 0
        probs = probs / sums.clamp_min(1e-12)
        if not torch.all(valid):
            uniform = torch.full_like(probs, 1.0 / probs.size(dim))
            probs = torch.where(valid, probs, uniform)
        return probs

    @staticmethod
    def _safe_vis_name(value):
        value = os.path.splitext(os.path.basename(str(value)))[0]
        safe_value = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)
        return safe_value or "sample"

    def _feature_vis_epoch_tag(self, epoch):
        if epoch is None:
            return "epoch_unknown"
        try:
            return f"epoch_{int(epoch):03d}"
        except (TypeError, ValueError):
            return self._safe_vis_name(epoch)

    def _feature_vis_names(self, image_names, batch_size):
        if image_names is None:
            start = getattr(self, "vis_counter", 0)
            names = [f"sample_{start + idx:06d}" for idx in range(batch_size)]
            self.vis_counter = start + batch_size
            return names

        if isinstance(image_names, str):
            names = [image_names]
        else:
            try:
                names = list(image_names)
            except TypeError:
                names = [image_names]

        if len(names) == 1 and batch_size > 1:
            base_name = self._safe_vis_name(names[0])
            return [f"{base_name}_{idx:02d}" for idx in range(batch_size)]

        while len(names) < batch_size:
            names.append(f"sample_{len(names):06d}")

        return [self._safe_vis_name(names[idx]) for idx in range(batch_size)]

    def _plot_feature_map(self, feature_tensor, save_path, title):
        if feature_tensor is None:
            return
        feature_tensor = feature_tensor.detach().float()
        if feature_tensor.dim() == 4 and feature_tensor.size(0) == 1:
            feature_tensor = feature_tensor[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_feature_map(feature_tensor, save_path, title=title)

    @staticmethod
    def _feature_vis_batch_size(feature_maps):
        if isinstance(feature_maps, (list, tuple)):
            if not feature_maps:
                raise ValueError("feature map list is empty")
            return feature_maps[0].size(0)
        return feature_maps.size(0)

    @staticmethod
    def _feature_vis_levels(feature_maps):
        if isinstance(feature_maps, (list, tuple)):
            return list(enumerate(feature_maps))
        return [(None, feature_maps)]

    def _save_feature_visualizations(
        self,
        image_names,
        epoch,
        mfb_features,
        mfb_position_encodings,
        retrieved_dmb_features,
        sam2_semantic_features,
        unet_decoder_features,
    ):
        epoch_tag = self._feature_vis_epoch_tag(epoch)
        root_dir = os.path.join(self.feature_vis_dir, epoch_tag)
        names = self._feature_vis_names(image_names, self._feature_vis_batch_size(mfb_features))

        for batch_idx, image_name in enumerate(names):
            for level_idx, mfb_feature in self._feature_vis_levels(mfb_features):
                level_suffix = "" if level_idx in (None, 0) else f"_level{level_idx:02d}"
                self._plot_feature_map(
                    mfb_feature[batch_idx],
                    os.path.join(root_dir, "mfb", f"{image_name}_mfb{level_suffix}.png"),
                    title=f"MFB Output{level_suffix}",
                )
            if mfb_position_encodings is not None:
                for level_idx, position_encoding in self._feature_vis_levels(mfb_position_encodings):
                    level_suffix = "" if level_idx in (None, 0) else f"_level{level_idx:02d}"
                    self._plot_feature_map(
                        position_encoding[batch_idx],
                        os.path.join(
                            root_dir,
                            "mfb_position_encodings",
                            f"{image_name}_mfb_position{level_suffix}.png",
                        ),
                        title=f"MFB Position Encoding{level_suffix}",
                    )
            self._plot_feature_map(
                sam2_semantic_features[batch_idx],
                os.path.join(root_dir, "sam2_semantic_stream", f"{image_name}_sam2_semantic_stream.png"),
                title="SAM2 Semantic Stream",
            )
            self._plot_feature_map(
                unet_decoder_features[batch_idx],
                os.path.join(root_dir, "unet_decoder", f"{image_name}_unet_decoder.png"),
                title="U-Net Style Decoder",
            )

            for rank, dmb_feature in enumerate(retrieved_dmb_features[batch_idx], start=1):
                self._plot_feature_map(
                    dmb_feature,
                    os.path.join(
                        root_dir,
                        "retrieved_dmb",
                        f"{image_name}_retrieved_dmb_top{rank:02d}.png",
                    ),
                    title=f"Retrieved DMB Top-{rank}",
                )

        saved_epochs = getattr(self, "_saved_all_memory_feature_epochs", set())
        if epoch_tag in saved_epochs:
            return

        memory_dir = os.path.join(root_dir, "all_dmb_memory")
        for memory_idx, (memory_feature, _, iou_score, _) in enumerate(self.memory_bank.memories):
            if isinstance(iou_score, torch.Tensor):
                iou_value = float(iou_score.detach().float().cpu().item())
            else:
                iou_value = float(iou_score)
            self._plot_feature_map(
                memory_feature,
                os.path.join(memory_dir, f"{epoch_tag}_dmb_memory_{memory_idx:02d}_iou_{iou_value:.4f}.png"),
                title=f"DMB Memory {memory_idx}",
            )

        saved_epochs.add(epoch_tag)
        self._saved_all_memory_feature_epochs = saved_epochs

    def forward(self, x, click=None, image_names=None, epoch=None):
        # backbone_out = self.image_encoder(x)
        backbone_out = self.model.forward_image(x) # net.forward_image(imgs)
        mfb_output_features = backbone_out.get(
            "mfb_output_features",
            backbone_out.get("backbone_fpn_ori", backbone_out["backbone_fpn"]),
        )
        # print("===========================")
        # print(f"mfb_output_features: {len(mfb_output_features[0][0].shape)}")  # 输出特征的形状
        mfb_position_encodings = backbone_out.get(
            "mfb_position_encodings",
            backbone_out.get("vision_pos_enc"),
        )
        _, vision_feats, vision_pos_embeds, _ = self.model._prepare_backbone_features(backbone_out)
        # self.model._prepare_backbone_features 特征全局视野权重计算
        #  在 _run_single_frame_inference->_get_image_feature 中 
        # vision_feats 
        # torch.Size([7744, 3, 32]) 88
        # torch.Size([1936, 3, 64]) 44
        # torch.Size([484, 3, 256]) 22
        _,_,y3 = backbone_out["vision_features"],backbone_out["vision_pos_enc"],backbone_out["backbone_fpn_ori"] #neck 暂时没啥用只是将连个多尺度融合然后平均，那么输出呢，会不会增加准确率
        x1, x2, x3, x4 = y3[0], y3[1], y3[2], y3[3]
        # torch.Size([12, 256, 88, 88]) x1
        # torch.Size([12, 256, 44, 44]) x2
        # torch.Size([12, 256, 22, 22]) x3
        # torch.Size([12, 256, 11, 11]) x4

        B = vision_feats[-1].size(1)  # batch size 
        retrieved_dmb_features = [[] for _ in range(B)]
        if len(self.memory_bank.memories) == 0: # 不用memory bank
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
            # vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, 128)).to(device="cuda")
            # vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, 128)).to(device="cuda")
        else:
                to_cat_memory_dynamic = []
                to_cat_memory_pos_dynamic = []
                to_cat_image_embed_dynamic = []
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 22, 22) 
                vision_feats_temp = vision_feats_temp.reshape(B, -1)
                #  -----------------------
                for b in range(B):# 每个批次的当前特征都需要进行更新
                        # Retrieve relevant memories for this batch item
                        # retrieved_memories 取回的是topK相似的记忆库
                        # 其中len(retrieved_memories[i]) ==  4 => 是 push 进去 4个元素 ，为什么是取回4个最相似的
                        #
                        retrieved_memories, similarities = self.memory_bank.retrieve(F.normalize(vision_feats_temp[b], p=2, dim=0),top_k=2)
                        if retrieved_memories:
                            # Apply attention weights based on similarity
                            weights = self._safe_probabilities(similarities, dim=0)
                            for i, (memory, pos_enc, _, image_emd) in enumerate(retrieved_memories):
                                # Add weighted memory features
                                memory_feature = torch.nan_to_num(memory.cuda(non_blocking=True))
                                memory_pos = torch.nan_to_num(pos_enc.cuda(non_blocking=True))
                                image_emd = torch.nan_to_num(image_emd.cuda(non_blocking=True))
                                to_cat_memory_dynamic.append(memory_feature.flatten(2).permute(2, 0, 1) * weights[i])
                                to_cat_memory_pos_dynamic.append(memory_pos.flatten(2).permute(2, 0, 1))
                                to_cat_image_embed_dynamic.append(image_emd)
                                if not self.training:
                                    retrieved_dmb_features[b].append(memory_feature.detach())

                if len(to_cat_memory_dynamic) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
                else:
                    memory_stack_ori = torch.stack(to_cat_memory_dynamic, dim=0) # 将4个选出最相似的记忆库进行堆叠
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos_dynamic, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed_dynamic, dim=0)
                    # vision_feats_temp 当前特征  memory_stack_ori 是记忆库中特征堆叠
                    image_embed_stack_ori = F.normalize(
                        torch.nan_to_num(image_embed_stack_ori.float()),
                        p=2,
                        dim=1,
                        eps=1e-6,
                    ) #标准化 图像255*22*22
                    vision_feats_temp = F.normalize(
                        torch.nan_to_num(vision_feats_temp.float()),
                        p=2,
                        dim=1,
                        eps=1e-6,
                    ) #当前输入图像的标准化
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t() #利用cos 相似性

                    similarity_scores = self._safe_probabilities(similarity_scores, dim=1)
                    #采样函数，从挑出的最相似的特种中再次进行采样
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                    vision_feats[-1] = self.model.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                        )
                
        feat_sizes = []
        for feat in vision_feats:
            H = W = int(torch.sqrt(torch.tensor(feat.size(0))))  # HW is first dimension
            feat_sizes.append((H, W))
        

        feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        # 此处怎么解决
        '''prompt encoder'''         
        with torch.no_grad():
            def _format_points(click_coords, click_labels=None):
                click_torch = torch.as_tensor(click_coords, dtype=torch.float, device=x.device)
                if click_torch.ndim == 1:
                    if click_torch.shape[0] != 2:
                        raise ValueError(f"Unsupported click coords shape: {tuple(click_torch.shape)}")
                    # [2] -> [1, 1, 2]
                    click_torch = click_torch.unsqueeze(0).unsqueeze(0)
                elif click_torch.ndim == 2:
                    if click_torch.shape[-1] != 2:
                        raise ValueError(
                            f"Unsupported click coords shape: {tuple(click_torch.shape)}. "
                            "Expecting point prompts with last dim = 2."
                        )
                    # [B,2] -> [B,1,2], or [N,2] with B=1 -> [1,N,2]
                    if click_torch.shape[0] == x.size(0):
                        click_torch = click_torch.unsqueeze(1)
                    else:
                        click_torch = click_torch.unsqueeze(0)
                elif click_torch.ndim == 3:
                    if click_torch.shape[-1] != 2:
                        raise ValueError(f"Unsupported click coords shape: {tuple(click_torch.shape)}")
                else:
                    raise ValueError(f"Unsupported click coords shape: {tuple(click_torch.shape)}")

                if click_labels is None:
                    click_label_torch = torch.ones(
                        (click_torch.size(0), click_torch.size(1)),
                        dtype=torch.int,
                        device=x.device
                    )
                else:
                    click_label_torch = torch.as_tensor(click_labels, dtype=torch.int, device=x.device)
                    if click_label_torch.ndim == 0:
                        click_label_torch = click_label_torch.unsqueeze(0).unsqueeze(0)
                    elif click_label_torch.ndim == 1:
                        if click_torch.size(0) == 1:
                            click_label_torch = click_label_torch.unsqueeze(0)
                        else:
                            click_label_torch = click_label_torch.unsqueeze(1)
                    elif click_label_torch.ndim != 2:
                        raise ValueError(f"Unsupported click label shape: {tuple(click_label_torch.shape)}")

                    # Align label shape with coords if needed.
                    if click_label_torch.shape[0] != click_torch.shape[0]:
                        if click_label_torch.shape[0] == 1:
                            click_label_torch = click_label_torch.repeat(click_torch.shape[0], 1)
                        else:
                            raise ValueError(
                                f"Batch mismatch between click coords {tuple(click_torch.shape)} "
                                f"and labels {tuple(click_label_torch.shape)}"
                            )
                    if click_label_torch.shape[1] != click_torch.shape[1]:
                        if click_label_torch.shape[1] == 1:
                            click_label_torch = click_label_torch.repeat(1, click_torch.shape[1])
                        else:
                            raise ValueError(
                                f"Point count mismatch between click coords {tuple(click_torch.shape)} "
                                f"and labels {tuple(click_label_torch.shape)}"
                            )
                return click_torch, click_label_torch

            def _format_boxes(boxes):
                box_torch = torch.as_tensor(boxes, dtype=torch.float, device=x.device)
                if box_torch.ndim == 1:
                    if box_torch.shape[0] != 4:
                        raise ValueError(f"Unsupported box shape: {tuple(box_torch.shape)}")
                    box_torch = box_torch.unsqueeze(0)
                elif box_torch.ndim == 2:
                    if box_torch.shape[-1] != 4:
                        raise ValueError(f"Unsupported box shape: {tuple(box_torch.shape)}")
                else:
                    raise ValueError(f"Unsupported box shape: {tuple(box_torch.shape)}")

                if box_torch.shape[0] != x.size(0):
                    if box_torch.shape[0] == 1:
                        box_torch = box_torch.repeat(x.size(0), 1)
                    else:
                        raise ValueError(
                            f"Batch mismatch between input batch {x.size(0)} and boxes {tuple(box_torch.shape)}"
                        )
                return box_torch

            click_points = None
            click_boxes = None
            if click is None:
                # Prompt-free branch: generate empty points to let SAM2 use its empty embeddings
                B = x.size(0)
                click_torch = torch.empty(B, 0, 2, dtype=torch.float, device=x.device)
                click_label_torch = torch.empty(B, 0, dtype=torch.int, device=x.device)
                click_points = (click_torch, click_label_torch)
            elif isinstance(click, dict):
                raw_boxes = click.get("boxes", click.get("bbox", None))
                raw_points = click.get("points", None)
                if raw_boxes is None and raw_points is None:
                    raise ValueError("Prompt dict must include at least one of: boxes/bbox/points")

                if raw_boxes is not None:
                    click_boxes = _format_boxes(raw_boxes)

                if raw_points is not None:
                    if isinstance(raw_points, (tuple, list)) and len(raw_points) == 2:
                        click_coords, click_labels = raw_points
                    else:
                        click_coords, click_labels = raw_points, None
                    click_points = _format_points(click_coords, click_labels)
            else:
                # Backward compatibility:
                # 1) click = coords
                # 2) click = (coords, labels)
                if isinstance(click, (tuple, list)) and len(click) == 2:
                    click_coords, click_labels = click
                else:
                    click_coords, click_labels = click, None
                click_points = _format_points(click_coords, click_labels)
            
            # print("===========================")
            # print(f"Click points shape: {click_points[0]}, Click labels shape: {click_points[1]}")
            # print("===========================")

            # Align bbox prompt path with SAM2ImagePredictor:
            # convert each box to two corner points with labels (2, 3),
            # then feed all sparse prompts via `points` only.
            if click_boxes is not None:
                box_coords = click_boxes.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=x.device)
                box_labels = box_labels.repeat(click_boxes.size(0), 1)
                if click_points is not None:
                    concat_coords = torch.cat([box_coords, click_points[0]], dim=1)
                    concat_labels = torch.cat([box_labels, click_points[1]], dim=1)
                    click_points = (concat_coords, concat_labels)
                else:
                    click_points = (box_coords, box_labels)

            se, de = self.model.sam_prompt_encoder(  #point  prompt
                points=click_points, #(coords_torch, labels_torch)
                boxes=None,
                masks=None,
            ) 

        '''train mask decoder'''      
        # prodict 
        low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = self.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False, # args.multimask_output if you want multiple masks
                repeat_image=False,  # the image is already batched
                high_res_features = high_res_feats
            )  
                # resize prediction
        pred = F.interpolate(low_res_multimasks,size=(x.size(2),x.size(3)))
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(x.size(2),x.size(3)),
                                            mode="bilinear", align_corners=False)

        # 内存机制的地方
        # 以下 x1 x2 x3 x4  需要相同的通道
        # 上采样的过程
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')

        # print("===========================")
        # print(f"High-Res Multimasks Shape: {high_res_multimasks.shape}")
        # print(f"UNet Output Shape: {out.shape}")
        # SAM-dominant residual fusion:
        # SAM2 provides the main semantic logits; the U-Net-style branch adds a bounded
        # residual for boundary/detail correction instead of directly averaging logits.
        sam_term = high_res_multimasks
        # Use a softened, detached SAM2 probability to build a routing gate.
        # Softening avoids saturating the gate when SAM logits are very large,
        # and max-pooling widens the gate around the SAM boundary band.
        sam_prob_for_gate = torch.sigmoid((sam_term / self.detail_gate_temperature).detach())
        sam_uncertainty = 1.0 - torch.abs(2.0 * sam_prob_for_gate - 1.0)
        boundary_gate = F.max_pool2d(
            sam_uncertainty,
            kernel_size=self.detail_boundary_kernel,
            stride=1,
            padding=self.detail_boundary_kernel // 2,
        )
        detail_gate = self.detail_gate_floor + (1.0 - self.detail_gate_floor) * boundary_gate
        # Actual learnable residual strength. Since tanh(out) is in [-1, 1],
        # unet_term is bounded by +/- detail_gain * detail_gate.
        detail_gain = self.max_detail_gain * torch.sigmoid(self.detail_gain_logit)
        unet_term = detail_gain * detail_gate * torch.tanh(out)
        opposite_scale_map = None
        opposite_ratio = None
        if self.etis_boundary_optimized_fusion_flag.item() > 0.5:
            # ETIS tends to suffer when the detail residual erases confident SAM2
            # foreground/background regions. Keep boundary corrections intact near
            # the zero-logit contour, but shrink residuals that oppose confident SAM2.
            confidence = (sam_term.detach().abs() / self.etis_confidence_margin.clamp_min(1e-6)).clamp(0.0, 1.0)
            opposite_mask = (unet_term * sam_term.detach()) < 0
            opposite_scale_map = 1.0 - (1.0 - self.etis_opposite_residual_scale) * confidence
            unet_term = torch.where(opposite_mask, unet_term * opposite_scale_map, unet_term)
            opposite_ratio = opposite_mask.float().mean()
        pr = sam_term + unet_term
        self._record_fusion_stats(
            sam_term,
            unet_term,
            pr,
            detail_gain=detail_gain,
            detail_gate=detail_gate,
            boundary_gate=boundary_gate,
            opposite_scale=opposite_scale_map,
            opposite_ratio=opposite_ratio,
        )
        # pr = high_res_multimasks  # 直接使用高分辨率的 SAM2 输出作为最终预测，跳过 UNet 的融合

        if self.memory_mask_source == "sam":
            memory_mask_logits = high_res_multimasks.detach()
        elif self.memory_mask_source == "blend":
            blend = self.memory_mask_blend
            memory_mask_logits = (blend * pr + (1 - blend) * high_res_multimasks).detach()
        else:
            memory_mask_logits = pr.detach()

        with torch.no_grad():
            memory_iou_predictions = iou_predictions.detach()
            '''memory encoder'''       
            # new caluculated memory features # memory_encoder  控制通道
            # based on mutiple scale ,before input ,the cannel is 256 
            # add by leiwb start
            # 32 64 256
            pix_channel = [ele.size(2) for ele in vision_feats]
            # self.conv_s2 = nn.Conv2d(pix_channel[-2], self.model.hidden_dim, kernel_size=1, stride=1,device=torch.device("cuda"))
            # self.conv_s3 = nn.Conv2d(pix_channel[-3], self.model.hidden_dim, kernel_size=1, stride=1,device=torch.device("cuda"))

            # # top-level feature, (HW)BC => BCHW
            for i in range(len(vision_feats)):
                vision_feats[i] = vision_feats[i].permute(1, 2, 0).view(B, pix_channel[i], *feat_sizes[i])
            
            # 
            # vision_feats[-2] = self.conv_s2(vision_feats[-2])
            # vision_feats[-3] = self.conv_s3(vision_feats[-3])
            vision_feats[-2] = self.conv_s2(vision_feats[-2])
            vision_feats[-3] = self.conv_s3(vision_feats[-3])
            # BCHW => (HW)BC
            vision_feats = [feat.flatten(2).permute(2, 0, 1) for feat in vision_feats]
            # add by leiwb end

            # self.model._encode_new_memory 经过编码将当前特征编码 =>记忆库中的权重
            # 将记忆权重进行放入后续记忆库中，输出记忆权重可以进行使用 多尺度采样融合，然后进行进入记忆库中
            maskmem_features, maskmem_pos_enc,memory_out_array = self.model._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=memory_mask_logits,
                is_mask_from_pts=True)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
            # memory_out_array[0]["vision_features"].shape torch.Size([12, 64, 22, 22])
            # memory_out_array[1]["vision_features"].shape torch.Size([12, 64, 44, 44])
            # memory_out_array[2]["vision_features"].shape torch.Size([12, 64, 88, 88])

            mm_44      = memory_out_array[1]["vision_features"]
            mmpos_44   = memory_out_array[1]["vision_pos_enc"]

            mm_88      = memory_out_array[2]["vision_features"]
            mmpos_88   = memory_out_array[2]["vision_pos_enc"]
    
            # 使用预定义的 MFB 模块
            # maskmem_features_mfb = self.feat_small(maskmem_features)
            # mm_44_mfb = self.feat_mid(mm_44)
            # mm_88_mfb = self.feat_large(mm_88)


            # mm_2 = self.down_2(mm_44_mfb)
            # mm_4 = self.down_4(mm_88_mfb)

            # current_mm  =  (maskmem_features_mfb+ mm_2 + mm_4)/3
            # current_mm  =  maskmem_features_mfb
            current_mm  =  maskmem_features

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=torch.device("cuda"), non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=torch.device("cuda"), non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if self.training:
                if len(self.memory_bank.memories)== 0:
                    for batch in range(maskmem_features.size(0)):
                        
                        self.memory_bank.update(
                                                (maskmem_features[batch].unsqueeze(0)).detach(),
                                                (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                memory_iou_predictions[batch, 0],
                                                image_embed[batch].reshape(-1).detach()
                                            )
                else:
                    for batch in range(maskmem_features.size(0)):

                        push_maskmem_features = self.scale_factor*maskmem_features[batch].unsqueeze(0) + (1-self.scale_factor)*current_mm[batch]
                        # print("===========================")
                        # print(push_maskmem_features.shape)torch.Size([1, 64, 22, 22])
                        self.memory_bank.update(push_maskmem_features, # 更新输出由self.model._encode_new_memory
                                                maskmem_pos_enc[batch].unsqueeze(0),
                                                memory_iou_predictions[batch, 0],
                                                image_embed[batch].reshape(-1)
                                                )
        # if not self.training:
        #     # 输出内存的容量和当前使用的记忆数量
        #     print(f"Memory Bank Size: {len(self.memory_bank.memories)} / {self.memory_bank.max_size}")
            
        
        if not self.training and self.feature_vis_enabled: # 只在测试/推理阶段按开关画图
            try:
                self._save_feature_visualizations(
                    image_names=image_names,
                    epoch=epoch,
                    mfb_features=mfb_output_features,
                    mfb_position_encodings=mfb_position_encodings,
                    retrieved_dmb_features=retrieved_dmb_features,
                    sam2_semantic_features=high_res_multimasks,
                    unet_decoder_features=out,
                )
            except Exception as e:
                print(f"Failed to save feature map: {e}")
        return pr, out1, out2, high_res_multimasks # 方案：Learnable Temperature Scaling 动态对齐尺度，融合 SAM2 的高分辨率输出和 UNet 的细节解码输出
        # return pr, pr, pr  # 直接使用 SAM2 的高分辨率输出作为最终预测，跳过 UNet 的融合

if __name__ == "__main__":
    with torch.no_grad():
        # 输出部分进行融合
        model = MMSAM2().cuda()
        x = torch.randn(2, 3, 352, 352).cuda()
        out, out1, out2, high_res_multimasks = model(x)
        print(out.shape, out1.shape, out2.shape, high_res_multimasks.shape)
        # SAM2UNet 中初始化:
        # memory1 = memory_forward(256, 256, self)  # 传入self以使用SAM2的memory attention
