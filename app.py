import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Page Config ---
st.set_page_config(page_title="MNIST Cyber Predictor", page_icon="💾", layout="centered")

# --- Custom Y2K / Matrix Green CSS + Animated Sides ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* Global background and text */
    .stApp {
        background-color: #0a0a0a;
        background-image: linear-gradient(rgba(0, 255, 65, 0.05) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(0, 255, 65, 0.05) 1px, transparent 1px);
        background-size: 20px 20px;
        font-family: 'VT323', monospace;
        color: #00ff41;
    }

    /* Bold Text styling */
    h1, h2, h3, p, span, div {
        font-family: 'VT323', monospace !important;
        color: #00ff41 !important;
        text-shadow: 0 0 8px #00ff41;
    }

    h1 {
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 0px;
    }

    /* --- ANIMATED SIDE DECORATIONS --- */
    .side-decoration {
        position: fixed;
        top: 0;
        bottom: 0;
        width: 100px;
        pointer-events: none; /* Lets you click through them */
        z-index: 0;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        font-size: 2.5rem;
        opacity: 0.6;
    }
    
    .left-side { left: 10px; }
    .right-side { right: 10px; }

    /* The floating animation */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    .float-1 { animation: float 4s ease-in-out infinite; }
    .float-2 { animation: float 5s ease-in-out infinite 1s; }
    .float-3 { animation: float 6s ease-in-out infinite 2s; }
    .float-4 { animation: float 4.5s ease-in-out infinite 0.5s; }

    /* Blinking cursor effect */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    .cursor { animation: blink 1s step-end infinite; }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(0, 255, 65, 0.05);
        border: 2px dashed #00ff41;
        border-radius: 5px;
        box-shadow: 0 0 10px #00ff41;
    }
    [data-testid="stFileUploader"] section { background-color: transparent; }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

<!-- Left Side Animated Icons -->
<div class="side-decoration left-side">
    <div class="float-1">💾</div>
    <div class="float-2">👾</div>
    <div class="float-3">🖥️</div>
    <div class="float-4">📁</div>
</div>

<!-- Right Side Animated Icons -->
<div class="side-decoration right-side">
    <div class="float-3">📟</div>
    <div class="float-1">👁️</div>
    <div class="float-4">💿</div>
    <div class="float-2">🔋</div>
</div>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>💾 MNIST CYBER-SCAN 💾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.5rem;'>UPLOAD DIGITAL IMAGE FOR NEURAL NETWORK PREDICTION<span class='cursor'>_</span></p>", unsafe_allow_html=True)

# --- Load Model ---
MODEL_PATH = "models/mnist_cnn.keras" 

@st.cache_resource(show_spinner="INITIALIZING NEURAL NET...")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"SYSTEM ERROR - MODEL FAILED TO LOAD: {e}")
    st.stop()

# --- Preprocessing ---
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = arr.reshape(1, 28, 28, 1)
    return arr, img

# --- UI ---
uploaded_file = st.file_uploader(">> INSERT IMAGE DATA STREAM (.png, .jpg) <<", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center;'>RAW INPUT 📁</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    arr, processed = preprocess(image)
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>PROCESSED GRID 🖨️</h3>", unsafe_allow_html=True)
        st.image(processed, use_container_width=True)

    # Predict
    with st.spinner("ANALYZING PIXELS..."):
        preds = model.predict(arr, verbose=0)[0]
        
    predicted_class = int(np.argmax(preds))
    confidence = float(preds[predicted_class])

    st.markdown("<hr style='border: 1px solid #00ff41; box-shadow: 0 0 5px #00ff41;'>", unsafe_allow_html=True)
    
    # Big Digital Result
    st.markdown(f"""
        <div style="text-align: center; border: 2px solid #00ff41; padding: 20px; box-shadow: 0 0 15px #00ff41; margin-top: 20px; background: rgba(0, 255, 65, 0.05);">
            <h2 style="font-size: 2rem; margin: 0;">>> IDENTIFIED DIGIT <<</h2>
            <h1 style="font-size: 9rem; margin: 0; text-shadow: 0 0 25px #00ff41; font-weight: bold;">{predicted_class}</h1>
            <p style="font-size: 1.8rem; margin: 0;">CONFIDENCE: {confidence*100:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show probabilities
    st.markdown("<h3>📊 PROBABILITY MATRIX</h3>", unsafe_allow_html=True)
    prob_dict = {str(i): float(preds[i]) for i in range(10)}
    st.bar_chart(prob_dict, color="#00ff41")

    st.markdown("<p style='text-align: center; margin-top: 30px;'>💾 DATA SAVED SUCCESSFULLY 💾</p>", unsafe_allow_html=True)