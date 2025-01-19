HƯỚNG DẪN SỬ DỤNG PHẦN MỀM PHÂN LOẠI HẠT CÀ PHÊ
================================================
**LƯU Ý: VÌ ẢNH ĐƯỢC TRAIN KHÁ ÍT ( CHỈ 18 ẢNH NÊN SẼ BỊ OVERFITTING NHƯNG ) NÊN NẾU TEST THÌ CÓ THỂ DÙNG ẢNH TRONG THƯ MỤC dataset\train\images ĐỂ XEM THỬ DEMO
1. CẤU TRÚC THƯ MỤC
-------------------
dataset/                  # Thư mục chứa dữ liệu training
├── data.yaml            # File cấu hình dataset
├── train/               # Dữ liệu training
│   ├── images/          # Ảnh training
│   └── labels/          # Labels training
└── valid/               # Dữ liệu validation
    ├── images/          # Ảnh validation
    └── labels/          # Labels validation

models/                   # Thư mục chứa model đã train
└── best.pt              # Model đã train

runs/                     # Thư mục chứa kết quả training
└── train/               # Kết quả của quá trình training
    └── coffee_detection/# Thông tin chi tiết về training

2. CÁC FILE CHÍNH
----------------
train.py                 # File dùng để train model
app.py                   # File chạy giao diện web để detect
detector.py              # File xử lý phần detect hạt cà phê

3. CÁCH SỬ DỤNG
--------------
a) Training model:
   python train.py --data đường_dẫn_tới_dataset [--epochs 100] [--batch 16] [--imgsz 640]
   Ví dụ: python train.py --data D:\LEARNING\Test\dataset

b) Chạy ứng dụng:
   streamlit run app.py

4. PHÂN LOẠI HẠT CÀ PHÊ
----------------------
- Hạt chưa chín (Xanh)  : Màu xanh, chưa đủ độ chín
- Hạt sắp chín (Cam)    : Màu cam, gần đạt độ chín
- Hạt chín (Đỏ)         : Màu đỏ, đã đạt độ chín tối ưu

5. YÊU CẦU HỆ THỐNG
------------------
- Python 3.8 trở lên
- Các thư viện cần thiết:
  pip install torch==2.1.0 torchvision==0.16.0
  pip install ultralytics==8.0.196
  pip install streamlit
  pip install pillow
  pip install opencv-python

6. LƯU Ý
--------
- Đảm bảo đã cài đặt đầy đủ các thư viện trước khi chạy
- Kiểm tra GPU nếu muốn tăng tốc độ training
- File model sẽ được lưu tại models/best.pt sau khi training
- Có thể điều chỉnh các tham số training trong train.py
- Có thể điều chỉnh ngưỡng confidence trong detector.py

7. THÔNG TIN THÊM
----------------
- Classes: ['xanh', 'cam', 'do']
- Kích thước ảnh mặc định: 640x640
- Batch size mặc định: 16
- Số epochs mặc định: 100

8. XỬ LÝ LỖI THƯỜNG GẶP
----------------------
- Nếu gặp lỗi CUDA: Thử cài đặt lại PyTorch
- Nếu không tìm thấy model: Kiểm tra đường dẫn models/best.pt
- Nếu lỗi thư viện: Cài đặt lại các thư viện theo phiên bản đã nêu

Liên hệ hỗ trợ: giahuyworkmail@gmail.com
