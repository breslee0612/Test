from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch
import argparse

class CoffeeTrainer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def check_dataset(self):
        """Kiểm tra dataset trước khi training"""
        # Kiểm tra data.yaml
        yaml_path = self.data_dir / 'data.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file data.yaml tại {yaml_path}")
            
        # Đọc và hiển thị thông tin data.yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            print("\n📊 Thông tin Dataset:")
            print(f"- Classes: {data.get('names', [])}")
            print(f"- Số lượng classes: {data.get('nc', 0)}")
            
        # Kiểm tra số lượng ảnh và labels
        train_imgs = len(list((self.data_dir / 'train' / 'images').glob('*.*')))
        train_labels = len(list((self.data_dir / 'train' / 'labels').glob('*.txt')))
        valid_imgs = len(list((self.data_dir / 'valid' / 'images').glob('*.*')))
        valid_labels = len(list((self.data_dir / 'valid' / 'labels').glob('*.txt')))
        
        print("\n📁 Số lượng files:")
        print(f"- Train: {train_imgs} ảnh, {train_labels} labels")
        print(f"- Valid: {valid_imgs} ảnh, {valid_labels} labels")
        
        return yaml_path

    def train(self, epochs=100, batch_size=16, img_size=640):
        """Training model"""
        try:
            # Tắt CUDA để tránh lỗi
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            device = 'cpu'
            print(f"\n🖥️ Sử dụng: {device.upper()}")
            
            # Kiểm tra dataset
            yaml_path = self.check_dataset()
            
            # Khởi tạo model
            model = YOLO('yolov8n.pt')
            print("\n🚀 Bắt đầu training...")
            
            # Training
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project='runs/train',
                name='coffee_detection',
                exist_ok=True,
                patience=50,
                save=True,
                pretrained=True,
                optimizer='auto',
                verbose=True,
                seed=42,
                # Data augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0
            )
            
            # Lưu model
            save_dir = Path('models')
            save_dir.mkdir(exist_ok=True)
            
            # Copy model từ thư mục training
            best_model = Path('runs/train/coffee_detection/weights/best.pt')
            if best_model.exists():
                import shutil
                shutil.copy2(best_model, save_dir / 'best.pt')
                print(f"\n✅ Đã lưu model tại: {save_dir}/best.pt")
            
            print("\n🎉 Training hoàn thành!")
            
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            raise e

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 cho Coffee Detection')
    parser.add_argument('--data', type=str, required=True, help='Đường dẫn tới thư mục dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Số epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước ảnh')
    parser.add_argument('--project', type=str, default='coffee_detection', help='Tên project')
    
    args = parser.parse_args()
    
    trainer = CoffeeTrainer(args.data)
    trainer.train(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)

if __name__ == "__main__":
    main()
