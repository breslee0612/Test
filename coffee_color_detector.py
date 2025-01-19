import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class CoffeeColorDetector:
    def __init__(self):
        # Tham số cho phát hiện hạt
        self.min_area = 100
        self.max_area = 5000
        
        # Ngưỡng màu cho phân loại (HSV)
        # Màu đỏ/nâu cho hạt chín
        self.ripe_lower = np.array([0, 50, 50])
        self.ripe_upper = np.array([20, 255, 255])
        # Màu xanh cho hạt chưa chín
        self.unripe_lower = np.array([35, 50, 50])
        self.unripe_upper = np.array([85, 255, 255])

    def enhance_image(self, image):
        # Tăng độ tương phản
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def classify_bean(self, image, mask):
        """Phân loại hạt cà phê dựa trên màu sắc."""
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Tạo mask cho vùng hạt
        bean_region = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # Đếm pixel cho mỗi loại màu
        ripe_mask = cv2.inRange(bean_region, self.ripe_lower, self.ripe_upper)
        unripe_mask = cv2.inRange(bean_region, self.unripe_lower, self.unripe_upper)
        
        ripe_pixels = cv2.countNonZero(ripe_mask)
        unripe_pixels = cv2.countNonZero(unripe_mask)
        
        # Phân loại dựa trên số lượng pixel
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return "unknown"
        
        ripe_ratio = ripe_pixels / total_pixels
        unripe_ratio = unripe_pixels / total_pixels
        
        if ripe_ratio > unripe_ratio and ripe_ratio > 0.3:
            return "ripe"
        elif unripe_ratio > ripe_ratio and unripe_ratio > 0.3:
            return "unripe"
        else:
            return "unknown"

    def detect_beans(self, image):
        # Tăng cường ảnh
        enhanced = self.enhance_image(image)
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Phân đoạn ảnh
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Morphology
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Tìm contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        beans = []
        height, width = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Tạo mask cho hạt hiện tại
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Phân loại hạt
                bean_type = self.classify_bean(image, mask)
                
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    beans.append((ellipse, bean_type))
        
        return beans

    def create_yolo_annotations(self, beans, image_shape):
        """Tạo annotations theo format YOLO."""
        height, width = image_shape[:2]
        annotations = []
        
        for bean, bean_type in beans:
            (x_center, y_center), (width_e, height_e), angle = bean
            
            # Chuẩn hóa về khoảng [0,1]
            x_norm = x_center / width
            y_norm = y_center / height
            w_norm = width_e / width
            h_norm = height_e / height
            
            # Class ID: 0 cho ripe, 1 cho unripe, 2 cho unknown
            class_id = 0 if bean_type == "ripe" else 1 if bean_type == "unripe" else 2
            
            annotations.append([class_id, x_norm, y_norm, w_norm, h_norm])
        
        return annotations

    def visualize_results(self, image, beans):
        """Hiển thị kết quả với màu khác nhau cho từng loại."""
        result = image.copy()
        colors = {
            "ripe": (0, 255, 0),     # Xanh lá cho hạt chín
            "unripe": (0, 0, 255),   # Đỏ cho hạt chưa chín
            "unknown": (128, 128, 128) # Xám cho không xác định
        }
        
        for bean, bean_type in beans:
            color = colors[bean_type]
            cv2.ellipse(result, bean, color, 2)
            
            # Thêm nhãn
            (x, y), _, _ = bean
            cv2.putText(result, bean_type, (int(x), int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result

def process_directory(input_dir, output_dir, visualize=True):
    detector = CoffeeColorDetector()
    
    # Tạo thư mục output
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Tạo file classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('ripe\nunripe\nunknown')
    
    # Xử lý từng ảnh
    image_files = list(Path(input_dir).rglob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Chia train/val
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    for split, files in [('train', train_files), ('val', val_files)]:
        for image_path in files:
            try:
                print(f"Đang xử lý: {image_path}")
                
                # Đọc ảnh
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Phát hiện và phân loại hạt
                beans = detector.detect_beans(image)
                
                # Tạo annotations
                annotations = detector.create_yolo_annotations(beans, image.shape)
                
                # Lưu ảnh và annotations
                image_name = image_path.name
                label_name = f"{image_path.stem}.txt"
                
                # Copy ảnh
                cv2.imwrite(
                    os.path.join(output_dir, 'images', split, image_name),
                    image
                )
                
                # Lưu annotations
                with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                    for ann in annotations:
                        f.write(f"{' '.join(map(str, ann))}\n")
                
                if visualize:
                    # Hiển thị kết quả
                    result = detector.visualize_results(image, beans)
                    cv2.imshow('Detected Coffee Beans', result)
                    key = cv2.waitKey(0)
                    if key == 27:  # ESC
                        break
                
                print(f"Đã xử lý thành công: {image_path}")
                
            except Exception as e:
                print(f"Lỗi khi xử lý {image_path}: {str(e)}")
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Tạo file data.yaml
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: 3  # number of classes
names: ['ripe', 'unripe', 'unknown']  # class names
    """
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tạo dataset YOLO cho phân loại hạt cà phê')
    parser.add_argument('--input', type=str, required=True, help='Thư mục chứa ảnh input')
    parser.add_argument('--output', type=str, required=True, help='Thư mục output cho dataset')
    parser.add_argument('--no-viz', action='store_true', help='Tắt visualization')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, not args.no_viz) 