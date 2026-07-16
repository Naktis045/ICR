import os
# Suppress TF warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from datasets import load_dataset
from tensorflow.keras.layers import Layer, Dense, Embedding, LayerNormalization, MultiHeadAttention, Flatten

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ViT Handwriting Recognition",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS for modern styling 
st.markdown("""
    <style>
    .main-title { font-size: 2.8rem; font-weight: 700; color: #1E88E5; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 1.2rem; text-align: center; color: #555; margin-bottom: 40px; }
    .prediction-box { background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✍️ Vision Transformer Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A custom ViT model trained on the IAM dataset to recognize handwritten text</div>', unsafe_allow_html=True)

# ==========================================
# 1. VIT MODEL CLASS DEFINITION (Must match your training structure)
# ==========================================
class Patch_Encoder(Layer):
    def __init__(self, num_patches, hidden_size, patch_size=16, **kwargs):
        super(Patch_Encoder, self).__init__(name='patch_encoder', **kwargs)
        self.linear = Dense(hidden_size)
        self.positional_embedding = Embedding(input_dim=num_patches, output_dim=hidden_size)
        self.num_patches = num_patches
        self.patch_size = patch_size

    def call(self, x):
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, (batch_size, self.num_patches, -1))
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        output = self.linear(patches) + self.positional_embedding(positions)
        return output

class Transformer_Encoder(Layer):
    def __init__(self, num_heads, hidden_size, **kwargs):
        super(Transformer_Encoder, self).__init__(**kwargs)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.dense_1 = Dense(hidden_size, activation=tf.nn.gelu)
        self.dense_2 = Dense(hidden_size, activation=tf.nn.gelu)

    def call(self, input_tensor):
        x_1 = self.layer_norm_1(input_tensor)
        x_1 = self.multi_head_attention(x_1, x_1)
        x_1 = x_1 + input_tensor

        x_2 = self.layer_norm_2(x_1)
        x_2 = self.dense_1(x_2)
        output = self.dense_2(x_2)
        output = output + x_1
        return output

class VIT(tf.keras.Model):
    def __init__(self, num_heads, hidden_size, num_patches, num_dense_units, num_layers, num_classes, **kwargs):
        super(VIT, self).__init__(name="VIT", **kwargs)
        self.num_layers = num_layers
        self.patch_encoder = Patch_Encoder(num_patches, hidden_size)
        self.tr_encoders = [Transformer_Encoder(num_heads, hidden_size, name=f"transformer_encoder_{i}") for i in range(num_layers)]
        self.dense_1 = Dense(num_dense_units, tf.nn.gelu) 
        self.dense_2 = Dense(num_dense_units, tf.nn.gelu) 
        self.dense_3 = Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.patch_encoder(x)
        for i in range(self.num_layers):
            x = self.tr_encoders[i](x)
        x = Flatten()(x)
        x = self.dense_1(x) 
        x = self.dense_2(x)
        return self.dense_3(x)

# ==========================================
# 2. DATA LOADING & MODEL CACHING
# ==========================================
@st.cache_resource
def load_vocabulary():
    """Extracts vocabulary and class settings dynamically to align labels."""
    hf_dataset = load_dataset("Teklia/IAM-line")
    all_texts = list(hf_dataset['train']['text']) + list(hf_dataset['test']['text'])
    vocab = sorted(list(set([text[0] for text in all_texts if len(text) > 0])))
    # Include background class matching training format
    labels = ["_"] + vocab 
    return labels

@st.cache_resource
def load_vit_model(num_classes):
    """Instantiates the ViT architecture and loads trained weights."""
    try:
        # Re-build matching your specific hyperparameters
        vit_model = VIT(
            num_heads=8, 
            hidden_size=768, 
            num_patches=256, 
            num_layers=4, 
            num_dense_units=1024,
            num_classes=num_classes
        )
        # Compile Model
        vit_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        # Build states using dummy inputs
        dummy_input = tf.zeros((1, 256, 256, 3))
        _ = vit_model(dummy_input)
        
        # Load weights
        vit_model.load_weights('ViT_model_weights.h5')
        return vit_model
    except Exception as e:
        st.error(f"Could not initialize model: {e}")
        return None

# Load vocab and build model
VIT_LABELS = load_vocabulary()
num_classes_total = len(VIT_LABELS)
model = load_vit_model(num_classes=num_classes_total)

# ==========================================
# 3. USER INPUT INTERFACE
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Methods")
    input_mode = st.radio("Choose how to provide the character:", ("Draw on Canvas", "Upload an Image File"))
    
    final_image = None
    
    if input_mode == "Draw on Canvas":
        st.write("Draw a single character inside the box:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,  
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=256,
            width=256,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            # Keep standard RGB colorspace
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('RGB')
            if np.any(np.array(img) > 0):
                final_image = img

    else:
        uploaded_file = st.file_uploader("Upload image (JPG/PNG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            if st.checkbox("Invert Colors (Check if handwriting is dark and background is light)"):
                img = ImageOps.invert(img)
            st.image(img, caption="Original Uploaded Image", width=256)
            final_image = img

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
with col2:
    st.subheader("📊 Model Prediction")
    
    if final_image is not None and model is not None:
        # Resize to match training height and width requirements
        img_resized = final_image.resize((256, 256))
        
        # Convert PIL image straight into an normalized Numpy array [0, 1.0]
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Expand batch dimension (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        probabilities = model.predict(img_array, verbose=0)[0]
        
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx] * 100
        predicted_char = VIT_LABELS[best_class_idx]
        
        # Render main predictions block
        st.markdown(f"""
            <div class="prediction-box">
                <p style="margin:0; font-size:1.1rem; color:#333;">Top ViT Prediction</p>
                <h1 style="margin:0; font-size:4rem; color:#2E7D32;">{predicted_char}</h1>
                <p style="margin:0; font-size:1rem; color:#555;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("### ")
        
        # Display top-5 distributions
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_chars = [VIT_LABELS[i] for i in top_5_idx]
        top_5_probs = [probabilities[i] for i in top_5_idx]
        
        fig = go.Figure(go.Bar(
            x=top_5_probs,
            y=top_5_chars,
            orientation='h',
            marker_color='#4CAF50',
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
        st.info("Awaiting input data. Draw inside the canvas or upload an image on the left to display ViT inferences.")