import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO

# Load your pre-trained model
model = tf.keras.models.load_model("best_model.keras")

# Function to preprocess and predict on a single image
def classify_image(img, model, target_size=(256, 256)):
    # Preprocess the image
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    preds = model.predict(img_array)
    pred_prob = preds[0][0]  # Extract the probability for "Dog"

    # Classification logic
    label = "Dog" if pred_prob > 0.5 else "Cat"
    return label

# Streamlit app layout
st.set_page_config(page_title="Cat & Dog Classifier with CNN", layout="wide")
st.title("Cat & Dog Classifier with CNN")

# Header section
st.markdown("""
    <div style="background-color:#007bff; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color:white;">Upload One or More Images for Classification</h2>
        <p style="color:white;">Please upload Cat or Dog images for classification.</p>
    </div>
""", unsafe_allow_html=True)

# Image upload form
uploaded_files = st.file_uploader("Choose files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Create a session state variable to track uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Update session state when new files are uploaded
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Display images and classifications
if st.session_state.uploaded_files:
    st.markdown("""
        <div style="border: 2px solid #007bff; border-radius: 10px; padding: 10px; background-color: #f1f1f1; text-align: center;">
            <h3 style="color: #007bff; font-size: 24px;">Classification Results</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create 3 columns for displaying images in rows
    cols = st.columns(3)

    for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
        # Save the uploaded file temporarily
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
            temp_file_path = f.name

        # Load image for classification
        img = image.load_img(temp_file_path, target_size=(256, 256))

        # Get classification result from model
        label = classify_image(img, model)

        # Convert image to base64 for display
        img_base64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

        # Determine which column to use
        col_idx = idx % 3  # Cycles between 0, 1, 2
        with cols[col_idx]:
            # Display the image and the classification text in the same box
            st.markdown(f"""
                <div style="border: 2px solid #007bff; padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center;">
                    <img src="data:image/jpeg;base64,{img_base64}" width="200" />
                    <p style="font-size: 16px; margin-top: 10px; color: #007bff;"><strong>Given image is classified as: {label}</strong></p>
                </div>
            """, unsafe_allow_html=True)

    
    # Clear images button after classifications
    # Clear images button after classifications
    if st.button("Clear Images"):
        st.session_state.uploaded_files = []  # Clear uploaded files from session state
        st.query_params.clear()
        
        
# Footer
footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: gray;
        background-color: #333;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Developed by Sheema Masood | Powered by Streamlit</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
