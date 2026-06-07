#!/bin/bash
# train_polyp.sh
# 功能：针对Polyp息肉检测数据集启动训练。
# 作用：设置环境变量，调用train.py脚本，配置数据集路径和多个验证集列表。

CUDA_VISIBLE_DEVICES="0" \
python train.py \
--exp_name "Polyp" \
--data_path "../data/Polyp" \
--valid_list CVC-300 CVC-ClinicDB ETIS-LaribPolypDB Kvasir \
--hiera_path "../data/sam2.pt" \
--save_path "./logs" \
--epoch 500 \
--valid_interval 1 \
--lr 0.001 \
--batch_size 12 \
--weight_decay 5e-4
