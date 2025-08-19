import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import pandas as pd
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Thiết lập logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Sử dụng thiết bị: {device}")

class CellClassifier:
    def __init__(self, cellpose_results_dir, original_images_dir, output_dir, num_classes=2):
        """
        Khởi tạo Cell Classifier
        
        Parameters:
        - cellpose_results_dir: thư mục chứa kết quả phân đoạn từ Cellpose
        - original_images_dir: thư mục chứa ảnh gốc
        - output_dir: thư mục lưu kết quả
        - num_classes: số lượng lớp cần phân loại
        """
        self.cellpose_results_dir = Path(cellpose_results_dir)
        self.original_images_dir = Path(original_images_dir)
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        
        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Các tham số cho trích xuất tế bào
        self.cell_size = (64, 64)  # Kích thước tế bào trích xuất
        
        # Danh sách cho các tế bào đã trích xuất và nhãn
        self.cells = []
        self.labels = []

        # Transforms cho dữ liệu huấn luyện/đánh giá
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_cells_from_image(self, image_path, mask_path, annotated_labels=None):
        """
        Trích xuất các tế bào riêng lẻ từ ảnh dựa trên mask phân đoạn
        
        Parameters:
        - image_path: đường dẫn đến ảnh gốc
        - mask_path: đường dẫn đến mask phân đoạn từ Cellpose
        - annotated_labels: từ điển chứa nhãn đã được gán cho từng tế bào (nếu có)
                           {cell_id: label}
        
        Returns:
        - cell_images: danh sách các ảnh tế bào đã trích xuất
        - cell_labels: danh sách nhãn tương ứng (nếu có)
        """
        try:
            # Đọc ảnh và mask
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Không thể đọc mask: {mask_path}")
            
            # Đảm bảo kích thước khớp nhau
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Kích thước không khớp: image {image.shape}, mask {mask.shape}")
            
            # Gán nhãn cho từng khu vực trong mask
            labeled_mask = measure.label(mask > 0)
            regions = measure.regionprops(labeled_mask)
            
            cell_images = []
            cell_labels = []
            cell_ids = []
            
            # Trích xuất từng tế bào
            for i, region in enumerate(regions):
                # Bỏ qua các vùng quá nhỏ
                if region.area < 100:  # Có thể điều chỉnh ngưỡng
                    continue
                
                # Lấy bounding box
                minr, minc, maxr, maxc = region.bbox
                
                # Mở rộng bounding box
                padding = 10
                minr = max(0, minr - padding)
                minc = max(0, minc - padding)
                maxr = min(image.shape[0], maxr + padding)
                maxc = min(image.shape[1], maxc + padding)
                
                # Tạo mask cho tế bào này
                cell_mask = labeled_mask == region.label
                cell_mask = cell_mask[minr:maxr, minc:maxc]
                
                # Cắt ảnh theo bounding box
                cell_image = image[minr:maxr, minc:maxc].copy()
                
                # Áp dụng mask để chỉ lấy phần tế bào
                for c in range(3):  # Cho mỗi kênh màu
                    bg_mask = ~cell_mask
                    cell_image[:, :, c][bg_mask] = 0  # hoặc giá trị nền khác
                
                # Thay đổi kích thước ảnh tế bào để có kích thước cố định
                cell_image = cv2.resize(cell_image, self.cell_size, interpolation=cv2.INTER_AREA)
                
                # Chuyển đổi BGR -> RGB (OpenCV đọc ảnh là BGR)
                cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
                
                # Thêm vào danh sách
                cell_images.append(cell_image)
                cell_ids.append(region.label)
                
                # Thêm nhãn nếu có
                if annotated_labels and region.label in annotated_labels:
                    cell_labels.append(annotated_labels[region.label])
                else:
                    # Nếu không có nhãn, gán -1 hoặc giá trị mặc định khác
                    cell_labels.append(-1)
            
            return cell_images, cell_labels, cell_ids
            
        except Exception as e:
            logging.error(f"Lỗi khi trích xuất tế bào từ {image_path}: {str(e)}")
            return [], [], []
    
    def process_all_images(self, annotations_path=None):
        """
        Xử lý tất cả các ảnh và trích xuất tế bào
        
        Parameters:
        - annotations_path: đường dẫn đến file chứa thông tin gán nhãn (nếu có)
        """
        # Tìm tất cả các mask files đã được tạo bởi Cellpose
        mask_files = list(self.cellpose_results_dir.glob('*mask*.png'))
        annotated_labels = {}
        
        # Đọc annotations nếu có
        if annotations_path and Path(annotations_path).exists():
            try:
                annotations_df = pd.read_csv(annotations_path)
                # Giả định format: image_name, cell_id, label
                for _, row in annotations_df.iterrows():
                    image_name = row['image_name']
                    cell_id = row['cell_id']
                    label = row['label']
                    
                    if image_name not in annotated_labels:
                        annotated_labels[image_name] = {}
                    
                    annotated_labels[image_name][cell_id] = label
            except Exception as e:
                logging.error(f"Lỗi khi đọc file annotations: {str(e)}")
        
        logging.info(f"Tìm thấy {len(mask_files)} mask files từ Cellpose")
        
        # Xử lý từng ảnh
        for mask_path in tqdm(mask_files, desc="Trích xuất tế bào"):
            try:
                # Tìm tên ảnh tương ứng từ tên mask
                # Giả định format: masks_original_image_name.png
                mask_name = mask_path.name
                if mask_name.startswith("masks_"):
                    image_name = mask_name[6:]  # Bỏ "masks_" ở đầu
                else:
                    image_name = mask_name
                
                # Tìm ảnh gốc
                image_path = self.original_images_dir / image_name
                if not image_path.exists():
                    logging.warning(f"Không tìm thấy ảnh gốc: {image_path}")
                    continue
                
                # Lấy annotations cho ảnh này nếu có
                img_annotations = annotated_labels.get(image_name, {})
                
                # Trích xuất tế bào
                cell_images, cell_labels, cell_ids = self.extract_cells_from_image(
                    image_path, mask_path, img_annotations
                )
                
                # Thêm vào danh sách chung
                self.cells.extend(cell_images)
                self.labels.extend(cell_labels)
                
                # Lưu các tế bào đã trích xuất (nếu cần)
                self._save_extracted_cells(image_name, cell_images, cell_labels, cell_ids)
                
            except Exception as e:
                logging.error(f"Lỗi khi xử lý mask {mask_path}: {str(e)}")
                continue
        
        logging.info(f"Đã trích xuất tổng cộng {len(self.cells)} tế bào")
        
    def _save_extracted_cells(self, image_name, cell_images, cell_labels, cell_ids):
        """Lưu các tế bào đã trích xuất vào thư mục"""
        # Tạo thư mục cho ảnh này
        image_out_dir = self.output_dir / "extracted_cells" / image_name.split('.')[0]
        image_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu từng tế bào
        for i, (cell, label, cell_id) in enumerate(zip(cell_images, cell_labels, cell_ids)):
            # Tên file: {tên_ảnh}_cell{cell_id}_label{label}.png
            cell_filename = f"{image_name.split('.')[0]}_cell{cell_id}_label{label}.png"
            # Chuyển RGB về BGR cho OpenCV
            cell_bgr = cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_out_dir / cell_filename), cell_bgr)
    
    def build_classification_model(self):
        """Xây dựng mô hình phân loại tế bào sử dụng PyTorch"""
        # Sử dụng mô hình pretrained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Đóng băng các lớp
        for param in model.parameters():
            param.requires_grad = False
        
        # Thay thế lớp fully connected cuối cùng
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Chuyển mô hình sang thiết bị phù hợp (GPU nếu có)
        model = model.to(device)
        
        return model
    
    class CellDataset(Dataset):
        """Dataset cho các tế bào"""
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.images)
            
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    def train_model(self, model=None, epochs=20, batch_size=32, validation_split=0.2):
        """Huấn luyện mô hình phân loại tế bào sử dụng PyTorch"""
        if len(self.cells) == 0 or len(self.labels) == 0:
            raise ValueError("Không có dữ liệu để huấn luyện. Hãy chạy process_all_images trước.")
        
        # Kiểm tra có đủ nhãn hay không
        labeled_indices = [i for i, label in enumerate(self.labels) if label != -1]
        if len(labeled_indices) == 0:
            raise ValueError("Không có nhãn cho dữ liệu. Cần cung cấp file annotations.")
        
        # Chọn chỉ những tế bào đã được gán nhãn
        X = np.array([self.cells[i] for i in labeled_indices])
        y = np.array([self.labels[i] for i in labeled_indices])
        
        # Chia dữ liệu
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Tạo datasets và dataloaders
        train_dataset = self.CellDataset(X_train, y_train, transform=self.transform)
        val_dataset = self.CellDataset(X_val, y_val, transform=self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Tạo mô hình nếu chưa có
        if model is None:
            model = self.build_classification_model()
        
        # Định nghĩa loss function và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Theo dõi metrics
        best_val_loss = float('inf')
        best_model_weights = None
        train_losses = []
        val_losses = []
        val_accs = []
        
        # Training loop
        for epoch in range(epochs):
            ju
            model.train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = correct / total
            
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
            logging.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Loss: {epoch_train_loss:.4f} - "
                       f"Val Loss: {epoch_val_loss:.4f} - "
                       f"Val Acc: {epoch_val_acc:.4f}")
            
            # Lưu mô hình tốt nhất
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 5:
                logging.info(f"Early stopping tại epoch {epoch+1}")
                break
        
        # Tải mô hình tốt nhất
        model.load_state_dict(best_model_weights)
        
        # Đánh giá mô hình với tập validation
        self._evaluate_model(model, val_loader)
        
        # Plot learning curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accs, label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png')
        plt.close()
        
        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
    
    def _evaluate_model(self, model, val_loader):
        """Đánh giá mô hình và vẽ confusion matrix"""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Báo cáo phân loại
        report = classification_report(all_labels, all_preds)
        print("Classification Report:")
        print(report)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Lưu báo cáo
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
    
    def predict(self, model, image_path, mask_path):
        """Dự đoán lớp cho các tế bào trong ảnh mới"""
        # Trích xuất tế bào
        cell_images, _, cell_ids = self.extract_cells_from_image(image_path, mask_path)
        
        if len(cell_images) == 0:
            logging.warning(f"Không tìm thấy tế bào nào trong {image_path}")
            return None
        
        # Tạo dataset và dataloader
        test_dataset = self.CellDataset(cell_images, [-1] * len(cell_images), transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Dự đoán
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(probs, 1)
                conf, _ = torch.max(probs, 1)
                
                predictions.extend(preds.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
        
        # Tạo kết quả với ID tế bào
        results = {
            'cell_id': cell_ids,
            'predicted_class': predictions,
            'confidence': confidences
        }
        
        # Hiển thị kết quả
        self._visualize_predictions(image_path, mask_path, results)
        
        return results
    
    def _visualize_predictions(self, image_path, mask_path, results):
        """Trực quan hóa kết quả dự đoán"""
        # Đọc ảnh và mask
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Tạo ảnh kết quả
        labeled_mask = measure.label(mask > 0)
        overlay = segmentation.mark_boundaries(image, labeled_mask, color=(1, 0, 0), outline_color=(1, 1, 1))
        
        # Vẽ các nhãn
        plt.figure(figsize=(16, 12))
        plt.imshow(overlay)
        
        # Thêm text cho từng tế bào
        for i, (cell_id, pred_class, conf) in enumerate(zip(
            results['cell_id'], results['predicted_class'], results['confidence']
        )):
            # Tìm centroid của tế bào
            region = measure.regionprops(labeled_mask == cell_id)[0]
            y, x = region.centroid
            
            # Hiển thị class và confidence
            plt.text(x, y, f"Class {pred_class}\n{conf:.2f}", 
                     color='white', fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle="round", fc="black", alpha=0.7))
        
        plt.title(f"Dự đoán phân loại tế bào - {Path(image_path).name}")
        plt.axis('off')
        
        # Lưu kết quả
        output_path = self.output_dir / f"prediction_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Đã lưu kết quả dự đoán tại {output_path}")


# Ví dụ cách sử dụng
if __name__ == "__main__":
    # Cấu hình đường dẫn
    root_dir = Path(os.getcwd())
    cellpose_results_dir = root_dir / "output_cellpose"
    original_images_dir = root_dir / "Data/Labeling/P01_images"
    output_dir = root_dir / "output_classification"
    
    # Tạo instance
    classifier = CellClassifier(
        cellpose_results_dir=cellpose_results_dir,
        original_images_dir=original_images_dir,
        output_dir=output_dir,
        num_classes=2  # Tùy thuộc vào bài toán của bạn (ví dụ: 2 cho phân loại nhị phân)
    )
    
    # Xử lý các ảnh và trích xuất tế bào
    # Nếu bạn có file annotations, hãy cung cấp đường dẫn
    annotations_path = None  # hoặc Path("path/to/annotations.csv")
    classifier.process_all_images(annotations_path)
    
    # Huấn luyện mô hình (nếu có annotations)
    if annotations_path is not None:
        model, history = classifier.train_model(epochs=20, batch_size=32)
        
        # Lưu mô hình
        torch.save(model.state_dict(), output_dir / "cell_classifier_model.pth")
        
        # Dự đoán trên ảnh mới
        test_image_path = original_images_dir / "test_image.png"
        test_mask_path = cellpose_results_dir / "masks_test_image.png"
        
        if test_image_path.exists() and test_mask_path.exists():
            classifier.predict(model, test_image_path, test_mask_path)
    else:
        logging.info("Không có file annotations. Bạn cần cung cấp nhãn để huấn luyện mô hình.") 