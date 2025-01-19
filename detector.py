from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import warnings
import os
import io
from pathlib import Path

warnings.filterwarnings('ignore')  # Bỏ qua các cảnh báo

class CoffeeDetector:
    def __init__(self, model_path):
        # Tắt CUDA và cấu hình torch để tránh lỗi
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        # Tải model với device='cpu' để tránh vấn đề CUDA
        self.model = YOLO(model_path)
        self.model.to('cpu')
        
        # Định nghĩa các class và màu sắc
        self.classes = {
            0: 'do',    # Hạt chín
            1: 'xanh',  # Hạt chưa chín
            2: 'cam'    # Hạt sắp chín
        }
    
    def detect_and_count(self, image):
        try:
            results = self.model(image, conf=0.25)
            result = results[0]
            
            # In thông tin debug
            print(f"Detected boxes: {len(result.boxes)}")
            print(f"Confidence scores: {[round(float(box.conf), 2) for box in result.boxes]}")
            print(f"Detected classes: {[int(box.cls) for box in result.boxes]}")
            
            # Đếm số lượng cho từng loại
            cam_count = sum(1 for box in result.boxes if int(box.cls) == 2)   # class 2: hạt sắp chín (cam)
            xanh_count = sum(1 for box in result.boxes if int(box.cls) == 0)  # class 0: hạt chưa chín (xanh)
            do_count = sum(1 for box in result.boxes if int(box.cls) == 1)    # class 1: hạt chín (đỏ)
            
            # Vẽ bounding boxes
            plotted_image = result.plot()
            
            return {
                'cam': cam_count,    # hạt sắp chín
                'xanh': xanh_count,  # hạt chưa chín
                'do': do_count,      # hạt chín
                'total': len(result.boxes)
            }, Image.fromarray(plotted_image)
            
        except Exception as e:
            raise Exception(f"Lỗi trong quá trình detection: {str(e)}")
    
    def process_image(self, image_file):
        # Đọc ảnh từ file upload
        if isinstance(image_file, (str, Path)):
            img = Image.open(image_file)
        else:
            img = Image.open(io.BytesIO(image_file.read()))
            
        # Chuyển đổi ảnh sang định dạng phù hợp
        img_array = np.array(img)
        
        # Thực hiện detection với force_cpu=True
        results = self.model.predict(
            source=img_array,
            conf=0.25,
            device='cpu',
            verbose=False
        )
        
        # Khởi tạo bộ đếm
        counts = {
            'do': 0,
            'xanh': 0,
            'cam': 0,
            'total': 0
        }
        
        # Đếm số lượng mỗi loại
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                cls_name = self.classes[cls]
                counts[cls_name] += 1
                counts['total'] += 1
        
        # Lấy ảnh đã được annotate
        annotated_img = results[0].plot()
        
        return counts, annotated_img
