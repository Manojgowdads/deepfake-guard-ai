import os
import urllib.request
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Deepfake Guard", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    import os
    # Automatically finds the .h5 file in your GitHub folder
    possible_models = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not possible_models:
        st.error("🚨 ERROR: No .h5 model file found on GitHub!")
        return None
    
    model_path = possible_models[0]
    return tf.keras.models.load_model(model_path)
@st.cache_resource
def load_face_cascade():
    # 1. Point directly to the official OpenCV Github
    xml_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    
    # 2. Create a safe temporary spot on your PC
    temp_xml = os.path.join(tempfile.gettempdir(), 'haarcascade.xml')
    
    # 3. Auto-download a fresh, uncorrupted copy if it's not there
    if not os.path.exists(temp_xml):
        urllib.request.urlretrieve(xml_url, temp_xml)
        
    # 4. Load it!
    return cv2.CascadeClassifier(temp_xml)

model = load_model()
face_cascade = load_face_cascade()

# Clean Startup Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/8652/8652254.png", width=80) # Shield icon
with col2:
    st.title("Deepfake Guard AI 🛡️")
    st.markdown("**Advanced Neural Network & Facial Recognition Gatekeeper**")

st.markdown("---")

def analyze_face(frame_array):
    gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0: return None 
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_crop = frame_array[y:y+h, x:x+w]
    img_resized = cv2.resize(face_crop, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_ready = np.expand_dims(img_normalized, axis=0)
    return model.predict(img_ready)[0][0], face_crop

# Main UI layout
upload_col, result_col = st.columns(2)

with upload_col:
    st.subheader("1. Upload Media")
    uploaded_file = st.file_uploader("Upload Image or Video (.mp4, .jpg, .png)", type=["jpg", "jpeg", "png", "mp4"])

with result_col:
    st.subheader("2. AI Analysis")
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # --- IMAGE PROCESSING ---
        if file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file)
            st.image(image, caption='Source Media', width=250)
            
            with st.spinner("Scanning for human faces..."):
                img_array = np.array(image.convert('RGB'))
                result = analyze_face(img_array)
            
            if result is None:
                st.error("🛑 GATEKEEPER ERROR: No human face detected in this image.")
            else:
                confidence, face_crop = result
                st.image(face_crop, caption="Isolated Face", width=120)
                if confidence > 0.5:
                    st.error(f"🚨 **VERDICT: DEEPFAKE DETECTED** (Confidence: {confidence * 100:.2f}%)")
                else:
                    st.success(f"✅ **VERDICT: REAL HUMAN** (Confidence: {(1.0 - confidence) * 100:.2f}%)")
                    st.balloons()
                    
        # --- VIDEO PROCESSING ---
        elif file_extension == 'mp4':
            with st.spinner("Extracting and analyzing video frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                
                frame_count = 0
                predictions = []
                progress_bar = st.progress(0)
                
                while cap.isOpened() and frame_count < 30: 
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % 5 == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = analyze_face(frame_rgb)
                        if result is not None:
                            predictions.append(result[0])
                    frame_count += 1
                    progress_bar.progress(min(frame_count / 30.0, 1.0))
                cap.release()
            
            if len(predictions) == 0:
                st.error("🛑 GATEKEEPER ERROR: No human faces detected in video.")
            else:
                avg_confidence = sum(predictions) / len(predictions)
                st.info(f"Scanned {len(predictions)} valid facial frames.")
                if avg_confidence > 0.5:
                    st.error(f"🚨 **VERDICT: DEEPFAKE VIDEO** (Average Confidence: {avg_confidence * 100:.2f}%)")
                else:
                    st.success(f"✅ **VERDICT: REAL VIDEO** (Average Confidence: {(1.0 - avg_confidence) * 100:.2f}%)")
                    st.balloons()
