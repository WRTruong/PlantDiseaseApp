# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time
import os

# ====================== CẤU HÌNH ======================
st.set_page_config(
    page_title="Chẩn đoán bệnh cây trồng",
    page_icon="🌿",
    layout="wide"
)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "mobilenetv2_finetuned.keras")
    class_indices_path = os.path.join("model", "class_indices.json")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Đảo key-value: từ idx -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

# ====================== TIỀN XỬ LÝ ẢNH ======================
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ====================== GIAO DIỆN ======================
st.title("🌿 Hệ thống chẩn đoán bệnh cây trồng")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Tải lên ảnh lá cây")
    
    option = st.radio(
        "Chọn cách tải ảnh:",
        ["📁 Upload từ máy", "📷 Chụp ảnh trực tiếp"]
    )
    
    uploaded_file = None
    
    if option == "📁 Upload từ máy":
        uploaded_file = st.file_uploader(
            "Chọn ảnh lá cây",
            type=['jpg', 'jpeg', 'png']
        )
    else:
        uploaded_file = st.camera_input("Chụp ảnh lá cây")
    
    predict_btn = st.button(
        "🔍 Phân tích bệnh",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True
    )

with col2:
    st.subheader("📊 Kết quả phân tích")
    
    if uploaded_file and predict_btn:
        image = Image.open(uploaded_file)
        
        with st.spinner("🔄 Đang phân tích..."):
            time.sleep(1)  # Hiệu ứng loading
            
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            st.image(image, caption="Ảnh đã tải lên", width=300)
            
            disease_name = idx_to_class[predicted_idx]
            if "___" in disease_name:
                plant, disease = disease_name.split("___")
                formatted_name = f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"
            else:
                formatted_name = disease_name.replace("_", " ")
            
            if confidence > 0.8:
                st.success(f"**Kết luận:** {formatted_name}")
            elif confidence > 0.6:
                st.warning(f"**Kết luận:** {formatted_name}")
            else:
                st.error(f"**Kết luận:** {formatted_name}")
            
            st.progress(confidence)
            st.info(f"**Độ tin cậy:** {confidence*100:.2f}%")
            
            st.subheader("🎯 Top 3 dự đoán")
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                conf = float(predictions[0][idx])
                disease = idx_to_class[idx].replace("_", " ")
                st.write(f"{i+1}. {disease}: {conf*100:.1f}%")

# ====================== HƯỚNG DẪN ======================
st.markdown("---")
with st.expander("ℹ️ Hướng dẫn sử dụng"):
    st.markdown("""
    ### 📝 Hướng dẫn:
    1. Chụp/tải ảnh lá cây cần chẩn đoán
    2. Nhấn nút "Phân tích bệnh"
    3. Đọc kết quả và độ tin cậy
    
    ### 🌱 Mẹo chụp ảnh tốt:
    - Chụp lá bị bệnh rõ ràng
    - Ánh sáng đủ, không bị mờ
    - Lấy toàn bộ lá trong khung hình
    
    ### ⚠️ Lưu ý:
    - Hệ thống nhận diện được **38 loại bệnh** trên cây trồng
    - Kết quả mang tính tham khảo
    - Nên tham khảo ý kiến chuyên gia nông nghiệp
    """)

st.markdown("---")
st.caption("Đồ án AI - Nhận diện bệnh cây trồng | Sử dụng PlantVillage Dataset")

