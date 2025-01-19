import streamlit as st
from detector import CoffeeDetector
import os
import torch

# Cấu hình trang
st.set_page_config(
    page_title="Đếm Hạt Cà Phê",
    page_icon="☕",
    layout="wide"
)

# Thiết lập style với màu nền khác nhau cho từng loại
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { 
        width: 100%; 
        margin-top: 1rem; 
    }
    .metric-card-ripe {
        background-color: #FF6B6B;  /* Màu đỏ nhạt cho hạt chín */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-unripe {
        background-color: #98FB98;  /* Màu xanh lá nhạt cho hạt chưa chín */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-semi {
        background-color: #FFA07A;  /* Màu cam nhạt cho hạt sắp chín */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-total {
        background-color: #B0C4DE;  /* Màu xanh dương nhạt cho tổng số */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề
st.title("🔍 Ứng dụng Đếm Hạt Cà Phê")

# Thông tin GPU
if torch.cuda.is_available():
    device_info = f"🖥️ Đang sử dụng: CUDA - {torch.cuda.get_device_name(0)}"
else:
    device_info = "🖥️ Đang sử dụng: CPU (CUDA không khả dụng)"
st.info(device_info)

# Khởi tạo detector
@st.cache_resource
def load_model():
    try:
        # Kiểm tra best.pt trong thư mục models
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            # Nếu không tìm thấy, kiểm tra trong thư mục runs
            model_path = "runs/train/coffee_detection/weights/best.pt"
            if not os.path.exists(model_path):
                st.warning("⚠️ Không tìm thấy model tùy chỉnh, sử dụng model mặc định")
                model_path = "yolov8n.pt"
            else:
                st.success(f"Đã tìm thấy model tại: {model_path}")
                print(f"Loading model from: {model_path}")  # Debug log
        else:
            st.success(f"Đã tìm thấy model tại: {model_path}")
            print(f"Loading model from: {model_path}")  # Debug log
        
        # Kiểm tra file tồn tại và kích thước
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            print(f"Model file size: {file_size:.2f} MB")  # Debug log
        
        return CoffeeDetector(model_path)
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        print(f"Error details: {str(e)}")  # Debug log
        return None

detector = load_model()
if detector is None:
    st.error("Không thể khởi tạo model. Vui lòng kiểm tra lại cài đặt.")
    st.stop()

# Tạo 2 cột
col1, col2 = st.columns(2)

# Upload ảnh
with col1:
    st.subheader("📤 Tải lên ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh hạt cà phê", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
        
        if st.button("🔍 Đếm hạt cà phê"):
            with st.spinner("⏳ Đang xử lý..."):
                try:
                    # Thực hiện detection
                    counts, annotated_image = detector.process_image(uploaded_file)
                    
                    # Hiển thị kết quả trong cột 2
                    with col2:
                        st.subheader("📊 Kết quả phân tích")
                        
                        # Hiển thị số liệu
                        col_stats1, col_stats2 = st.columns(2)
                        
                        with col_stats1:
                            st.markdown("""
                            <div class="metric-card-unripe">
                                <h3>🟢 Hạt chưa chín (Xanh)</h3>
                                <h2>{} hạt</h2>
                            </div>
                            """.format(counts['cam']), unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="metric-card-semi">
                                <h3>🟠 Hạt sắp chín (Cam)</h3>
                                <h2>{} hạt</h2>
                            </div>
                            """.format(counts['do']), unsafe_allow_html=True)
                        
                        with col_stats2:
                            st.markdown("""
                            <div class="metric-card-ripe">
                                <h3>🔴 Hạt chín (Đỏ)</h3>
                                <h2>{} hạt</h2>
                            </div>
                            """.format(counts['xanh']), unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="metric-card-total">
                                <h3>📊 Tổng số hạt</h3>
                                <h2>{} hạt</h2>
                            </div>
                            """.format(counts['total']), unsafe_allow_html=True)
                        
                        # Hiển thị ảnh kết quả
                        st.image(annotated_image, caption="Kết quả phát hiện", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra: {str(e)}")

# Thông tin bên sidebar
st.sidebar.header("ℹ️ Thông tin")
st.sidebar.markdown("""
    ### Hướng dẫn sử dụng:
    1. Tải lên ảnh chứa hạt cà phê
    2. Nhấn nút "Đếm hạt cà phê"
    3. Xem kết quả phân tích
    
    ### Chú thích màu sắc:
    🟢 Xanh - Hạt chưa chín
    🟠 Cam - Hạt sắp chín
    🔴 Đỏ - Hạt chín
    
    ### Về ứng dụng:
    - Sử dụng YOLOv8 để phát hiện
    - Phân loại hạt theo độ chín
    - Xử lý trên GPU (nếu có)
""")
