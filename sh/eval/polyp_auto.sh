# polyp_auto.sh
# 功能：针对Polyp数据集的自动化评估脚本。
# 作用：遍历多个子集（CVC-300, etc.），调用test.py生成预测，再调用eval.py进行指标评估。

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Polyp/valid/CVC-300/images/" \
--test_gt_path "./data/Polyp/valid/CVC-300/masks/" \
--save_path "./data/Polyp/valid/CVC-300/preds/"

python eval.py \
--dataset_name "CVC-300" \
--pred_path "./data/Polyp/valid/CVC-300/preds/" \
--gt_path "./data/Polyp/valid/CVC-300/masks/"



CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Polyp/valid/CVC-ClinicDB/images/" \
--test_gt_path "./data/Polyp/valid/CVC-ClinicDB/masks/" \
--save_path "./data/Polyp/valid/CVC-ClinicDB/preds/"

python eval.py \
--dataset_name "CVC-ClinicDB" \
--pred_path "./data/Polyp/valid/CVC-ClinicDB/preds/" \
--gt_path "./data/Polyp/valid/CVC-ClinicDB/masks/"



CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Polyp/valid/ETIS-LaribPolypDB/images/" \
--test_gt_path "./data/Polyp/valid/ETIS-LaribPolypDB/masks/" \
--save_path "./data/Polyp/valid/ETIS-LaribPolypDB/preds/"

python eval.py \
--dataset_name "ETIS-LaribPolypDB" \
--pred_path "./data/Polyp/valid/ETIS-LaribPolypDB/preds/" \
--gt_path "./data/Polyp/valid/ETIS-LaribPolypDB/masks/"


CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Polyp/valid/Kvasir/images/" \
--test_gt_path "./data/Polyp/valid/Kvasir/masks/" \
--save_path "./data/Polyp/valid/Kvasir/preds/"

python eval.py \
--dataset_name "Kvasir" \
--pred_path "./data/Polyp/valid/Kvasir/preds/" \
--gt_path "./data/Polyp/valid/Kvasir/masks/"