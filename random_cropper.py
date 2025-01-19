import cv2
import numpy as np
import os
from pathlib import Path
import random

class RandomCropper:
    def __init__(self, crop_size=(640, 640), crops_per_image=6, overlap_threshold=0.3):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.overlap_threshold = overlap_threshold  # Ngưỡng chồng lấp cho phép giữa các crop
        
    def get_random_crop(self, image, previous_crops):
        """Tạo một crop ngẫu nhiên không chồng lấp quá nhiều với các crop trước."""
        height, width = image.shape[:2]
        crop_height, crop_width = self.crop_size
        
        # Đảm bảo kích thước crop không lớn hơn ảnh
        if crop_height > height or crop_width > width:
            raise ValueError("Kích thước crop lớn hơn kích thước ảnh!")
        
        max_attempts = 50  # Số lần thử tối đa để tìm vị trí phù hợp
        
        for _ in range(max_attempts):
            # Tạo vị trí ngẫu nhiên
            x = random.randint(0, width - crop_width)
            y = random.randint(0, height - crop_height)
            
            # Tạo rectangle cho crop hiện tại
            current_rect = (x, y, x + crop_width, y + crop_height)
            
            # Kiểm tra overlap với các crop trước
            valid_position = True
            for prev_crop in previous_crops:
                overlap_ratio = self.calculate_overlap(current_rect, prev_crop)
                if overlap_ratio > self.overlap_threshold:
                    valid_position = False
                    break
            
            if valid_position:
                return current_rect
        
        # Nếu không tìm được vị trí thỏa mãn sau max_attempts lần thử
        # Trả về vị trí cuối cùng và cảnh báo
        print("Warning: Không tìm được vị trí không chồng lấp sau nhiều lần thử")
        return current_rect

    def calculate_overlap(self, rect1, rect2):
        """Tính tỷ lệ chồng lấp giữa hai rectangle."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        # Tính diện tích giao nhau
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        rect1_area = (x2 - x1) * (y2 - y1)
        
        return intersection_area / rect1_area

    def crop_image(self, image_path, output_dir):
        """Tạo nhiều crop ngẫu nhiên từ một ảnh."""
        # Đọc ảnh
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Danh sách lưu các vị trí đã crop
        previous_crops = []
        crops_info = []
        
        base_name = Path(image_path).stem
        
        for i in range(self.crops_per_image):
            try:
                # Lấy vị trí crop mới
                crop_rect = self.get_random_crop(image, previous_crops)
                previous_crops.append(crop_rect)
                
                # Cắt ảnh
                x1, y1, x2, y2 = crop_rect
                cropped = image[y1:y2, x1:x2]
                
                # Tạo tên file mới
                new_name = f"{base_name}_crop{i+1}.jpg"
                output_path = os.path.join(output_dir, new_name)
                
                # Lưu ảnh
                cv2.imwrite(output_path, cropped)
                
                crops_info.append({
                    'path': output_path,
                    'rect': crop_rect
                })
                
            except Exception as e:
                print(f"Lỗi khi tạo crop {i+1}: {str(e)}")
        
        return crops_info

def process_directory(input_dir, output_dir, visualize=True):
    cropper = RandomCropper()
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).rglob(f'*{ext}'))
    
    total_crops = []
    
    for image_path in image_files:
        try:
            print(f"Đang xử lý: {image_path}")
            
            # Xử lý ảnh
            crops_info = cropper.crop_image(image_path, output_dir)
            total_crops.extend(crops_info)
            
            if visualize:
                # Hiển thị kết quả
                image = cv2.imread(str(image_path))
                vis_image = image.copy()
                
                # Vẽ các vùng đã crop
                for idx, crop_info in enumerate(crops_info):
                    x1, y1, x2, y2 = crop_info['rect']
                    color = (0, 255, 0)  # Màu xanh lá
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_image, f"Crop {idx+1}", (x1+5, y1+25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Hiển thị ảnh gốc với các vùng crop
                cv2.imshow('Original with crops', vis_image)
                
                # Hiển thị các crop
                crops = [cv2.imread(info['path']) for info in crops_info]
                rows = (len(crops) + 2) // 3  # Số hàng cần thiết
                display = np.zeros((rows*640, 1920, 3), dtype=np.uint8)
                
                for idx, crop in enumerate(crops):
                    i = idx // 3
                    j = idx % 3
                    display[i*640:(i+1)*640, j*640:(j+1)*640] = crop
                
                cv2.imshow('Crops', display)
                key = cv2.waitKey(0)
                if key == 27:  # ESC
                    break
            
            print(f"Đã xử lý thành công: {image_path}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý {image_path}: {str(e)}")
    
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"\nTổng số crop đã tạo: {len(total_crops)}")
    print(f"Các ảnh đã được lưu trong: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tạo crops ngẫu nhiên cho YOLO training')
    parser.add_argument('--input', type=str, required=True, help='Thư mục chứa ảnh input')
    parser.add_argument('--output', type=str, required=True, help='Thư mục output cho ảnh đã crop')
    parser.add_argument('--no-viz', action='store_true', help='Tắt visualization')
    parser.add_argument('--crops', type=int, default=6, help='Số lượng crop cho mỗi ảnh')
    parser.add_argument('--overlap', type=float, default=0.3, help='Ngưỡng chồng lấp cho phép (0-1)')
    
    args = parser.parse_args()
    
    # Cập nhật tham số
    RandomCropper.crops_per_image = args.crops
    RandomCropper.overlap_threshold = args.overlap
    
    process_directory(args.input, args.output, not args.no_viz) 