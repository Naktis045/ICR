import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Set up clean page config
st.set_page_config(
    page_title="Intelligent Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS for modern styling 
st.markdown("""
    <style>
    .main-title { font-size: 2.8rem; font-weight: 700; color: #1E88E5; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 1.2rem; text-align: center; color: #555; margin-bottom: 40px; }
    .prediction-box { background-color: #F0F4C3; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✍️ Intelligent Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An optimized CNN model trained on the EMNIST dataset to recognize handwritten characters</div>', unsafe_allow_html=True)

# 1. Cache the model load
@st.cache_resource
def load_emnist_model():
    try:
        return tf.keras.models.load_model('emnist_model.h5')
    except Exception as e:
        st.error("Could not find 'emnist_model.h5'. Make sure you have trained and saved the model in this directory.")
        return None

model = load_emnist_model()

# Mapping for EMNIST Balanced (47 classes: 0-9, A-Z, and select lowercase)
EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# Create layout columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Methods")
    input_mode = st.radio("Choose how to provide the character:", ("Draw on Canvas", "Upload an Image File"))
    
    final_image = None
    
    if input_mode == "Draw on Canvas":
        st.write("Draw a single character inside the box:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,  # Thicker brush helps match EMNIST line thickness
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            # Drop alpha channel, keep single channel grayscale
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            
            # Simple check to see if the user has actually drawn anything
            if np.any(np.array(img) > 0):
                final_image = img

    else:
        uploaded_file = st.file_uploader("Upload image (JPG/PNG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('L')
            # EMNIST characters are typically white text on a black background. 
            if st.checkbox("Invert Colors (Check if text is dark and background is light)"):
                img = ImageOps.invert(img)
            st.image(img, caption="Original Uploaded Image", width=200)
            final_image = img

with col2:
    st.subheader("📊 Model Prediction")
    
    if final_image is not None and model is not None:
        # 2. Preprocessing pipeline matching your tf.data logic
        img_resized = final_image.resize((28, 28))
        
        # CRITICAL FIX FOR EMNIST: Transpose the image (swap rows and columns) 
        # to match the native orientation format of the EMNIST training dataset.
        img_resized = img_resized.transpose(Image.TRANSPOSE)
        
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        
        # Format shape to match (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Handle Predictions securely
        logits = model.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()[0] # Turn raw output logits into clean probability distributions
        
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx] * 100
        predicted_char = EMNIST_LABELS[best_class_idx]
        
        # Display main metric card
        st.markdown(f"""
            <div class="prediction-box">
                <p style="margin:0; font-size:1.1rem; color:#444;">Top Prediction</p>
                <h1 style="margin:0; font-size:4rem; color:#1E88E5;">{predicted_char}</h1>
                <p style="margin:0; font-size:1rem; color:#666;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("### ")
        
        # 4. Display interactive Plotly chart for top 5 choices
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_chars = [EMNIST_LABELS[i] for i in top_5_idx]
        top_5_probs = [probabilities[i] for i in top_5_idx]
        
        fig = go.Figure(go.Bar(
            x=top_5_probs,
            y=top_5_chars,
            orientation='h',
            marker_color='#42A5F5',
            text=[f"{p*100:.1f}%" for p in top_5_probs],
            textposition='auto'
        ))
        fig.update_layout(
            title="Top 5 Most Likely Classes",
            xaxis_title="Probability Score",
            yaxis=dict(autorange="reversed"),
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Awaiting input data. Draw inside the box or upload a character image on the left column to get results.")