import os
import numpy as np
from cellpose import models, io
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure
import pandas as pd
import logging
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Cấu hình đường dẫn
try:
    root_dir = Path(os.getcwd())
    image_dir = root_dir / "Data/Labeling/P01_images"
    mask_dir = root_dir / "Data/Labeling/P01_masks"
    output_dir = root_dir / "output_cellpose"

    # Kiểm tra và tạo thư mục
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục mask: {mask_dir}")
    
    # Tạo thư mục output với parents=True
    output_dir.mkdir(parents=True, exist_ok=True)

except Exception as e:
    logging.error(f"Lỗi khi thiết lập đường dẫn: {str(e)}")
    raise

# 2. Khởi tạo model Cellpose
try:
    model = models.Cellpose(gpu=True, model_type='cyto2')
    logging.info("Đã khởi tạo model Cellpose thành công")
except Exception as e:
    logging.error(f"Lỗi khi khởi tạo model: {str(e)}")
    raise

# 3. Parameters
params = {
    'diameter': None,    # Tự động detect
    'flow_threshold': 0.8,
    # 'cellprob_threshold': 0.0,
    'channels': [0,0]    # Grayscale
}

# 4. Hàm tính toán metrics
def calculate_metrics(mask_true, mask_pred):
    """Tính toán các metrics chính"""
    try:
        results = {}
        
        # Chuyển đổi masks sang binary
        mask_true = mask_true > 0  # shape: (1024, 1024, 3)
        mask_pred = mask_pred > 0  # shape: (1024, 1024)
        
        # Xử lý sự khác biệt về kích thước
        # Nếu mask_true có 3 kênh, chuyển thành 1 kênh bằng cách lấy max hoặc kết hợp các kênh
        if len(mask_true.shape) == 3 and mask_true.shape[2] == 3:
            logging.info(f"Chuyển đổi mask_true từ {mask_true.shape} thành 2D mask")
            # Phương pháp 1: Lấy bất kỳ pixel nào có giá trị > 0 trong bất kỳ kênh nào
            mask_true_2d = np.any(mask_true, axis=2)
        else:
            mask_true_2d = mask_true
        
        # Kiểm tra mask rỗng
        if not np.any(mask_true_2d) or not np.any(mask_pred):
            logging.warning("Một trong các mask rỗng!")
            return {
                'iou': 0,
                'dice': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'n_true': 0,
                'n_pred': 0
            }
        
        # IoU
        intersection = np.logical_and(mask_true_2d, mask_pred).sum()
        union = np.logical_or(mask_true_2d, mask_pred).sum()
        results['iou'] = intersection / union if union > 0 else 0
        
        # Dice
        dice_denominator = mask_true_2d.sum() + mask_pred.sum()
        results['dice'] = 2 * intersection / dice_denominator if dice_denominator > 0 else 0
        
        # Precision và Recall
        results['precision'] = intersection / mask_pred.sum() if mask_pred.sum() > 0 else 0
        results['recall'] = intersection / mask_true_2d.sum() if mask_true_2d.sum() > 0 else 0
        
        # F1 Score
        if results['precision'] + results['recall'] > 0:
            results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
        else:
            results['f1'] = 0
        
        # Số lượng tế bào
        regions_true = measure.label(mask_true_2d)
        regions_pred = measure.label(mask_pred)
        results['n_true'] = len(np.unique(regions_true)) - 1  # Trừ background
        results['n_pred'] = len(np.unique(regions_pred)) - 1  # Trừ background
        
        # Log shape để debug
        logging.info(f"Kích thước mask_true: {mask_true.shape}, mask_pred: {mask_pred.shape}, mask_true_2d: {mask_true_2d.shape}")
        
        return results
    except Exception as e:
        logging.error(f"Lỗi khi tính toán metrics: {str(e)}")
        logging.error(f"Kích thước mask_true: {mask_true.shape}, mask_pred: {mask_pred.shape}")
        raise

# 5. Xử lý từng ảnh
all_metrics = []
image_files = list(image_dir.glob('*.png'))
mask_files = [mask_dir / f'masks_{img.name}' for img in image_files]

logging.info(f"Tìm thấy {len(image_files)} ảnh để xử lý")

for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files)):
    try:
        # Kiểm tra file tồn tại
        if not img_path.exists():
            logging.warning(f"Không tìm thấy file ảnh: {img_path}")
            continue
        if not mask_path.exists():
            logging.warning(f"Không tìm thấy file mask: {mask_path}")
            continue
            
        # Đọc ảnh và mask thật
        image = io.imread(img_path)
        mask_true = io.imread(mask_path)
        
        # Chạy Cellpose
        masks_pred, flows, styles, diams = model.eval(
            image,
            diameter=params['diameter'],
            flow_threshold=params['flow_threshold'],
            # cellprob_threshold=params['cellprob_threshold'],
            channels=params['channels']
        )
        
        # Tính toán metrics
        metrics = calculate_metrics(mask_true, masks_pred)
        metrics['image'] = img_path.name
        all_metrics.append(metrics)
        
        # Lưu kết quả visualization
        fig = plt.figure(figsize=(15,5))
        
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(mask_true, cmap='viridis')
        plt.title(f'Ground Truth\n({metrics["n_true"]} cells)')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(masks_pred, cmap='viridis')
        plt.title(f'Cellpose Prediction\n({metrics["n_pred"]} cells)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'result_{img_path.stem}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh {img_path.name}: {str(e)}")
        continue

# 6. Tổng hợp kết quả
try:
    if not all_metrics:
        raise ValueError("Không có metrics nào được tính toán!")
        
    df_metrics = pd.DataFrame(all_metrics)
    
    # Tính toán thống kê
    summary = df_metrics.describe()
    metrics_mean = df_metrics.mean(numeric_only=True)
    metrics_std = df_metrics.std(numeric_only=True)
    
    # In kết quả
    print("\nKết quả trung bình trên tất cả các ảnh:")
    for metric, value in metrics_mean.items():
        if metric != 'image':
            print(f"{metric}: {value:.4f} ± {metrics_std[metric]:.4f}")
    
    # Lưu kết quả
    df_metrics.to_csv(output_dir / "metrics_per_image.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv")
    
    logging.info("Đã hoàn thành xử lý và lưu kết quả")
    
except Exception as e:
    logging.error(f"Lỗi khi tổng hợp kết quả: {str(e)}")
    raise




