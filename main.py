
import cv2 as cv
import keras
import numpy as np
import streamlit as st

# Define CSS for light and dark modes
LIGHT_MODE_CSS = """
<style>
body {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-testid="stSidebar"] {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}
</style>
"""

DARK_MODE_CSS = """
<style>
body {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #2e2e2e !important;
    color: #ffffff !important;
}
</style>
"""

# Add a sidebar toggle for dark mode
mode = st.sidebar.radio("Choose your theme:", ["Light Mode", "Dark Mode"])

# Apply CSS dynamically based on the selected mode
if mode == "Dark Mode":
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# Title and Description
st.title("🌱 Plant Disease Detection")
st.write(
    """This leaf disease detection model uses deep learning techniques and transfer learning to identify plant diseases.
    It is trained on a dataset with 33 different types of leaf diseases."""
)
st.warning(
    "⚠️ Please upload leaf images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato. "
    "Other images may not yield accurate results."
)

# Load the pre-trained model
model = keras.models.load_model('Trained_model.h5')

# Define the list of class labels
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot',
    'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy',
    'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy',
    'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot',
    'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy',
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot',
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# File uploader for input images
uploaded_file = st.file_uploader("Upload a leaf image:")
if uploaded_file is not None:
    # Read the uploaded image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    
    # Preprocess the image
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    
    # Display the uploaded image
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
    
    # Make a prediction
    predictions = model.predict(normalized_image)
    confidence = predictions[0][np.argmax(predictions)] * 100
    
    # Display the result
    if confidence >= 80:
        st.success(f"✅ Result: {label_name[np.argmax(predictions)]} (Confidence: {confidence:.2f}%)")
    else:
        st.warning("⚠️ The model is not confident in its prediction. Please try another image.")
