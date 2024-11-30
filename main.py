
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
h1, h2, h3 {
    color: #2c3e50 !important;
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
h1, h2, h3 {
    color: #ecf0f1 !important;
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

# Add a header banner with a placeholder plant image
st.image(
    "https://via.placeholder.com/1000x300.png?text=Plant+Disease+Detection",
    caption="🌱 Transforming Agriculture with AI",
    use_column_width=True,
)

# Title and Description
st.title("🌱 Plant Disease Detection")
st.markdown(
    """
    ### Welcome to Plant Disease Detection App
    This tool uses cutting-edge **deep-learning techniques** to identify plant diseases 
    from leaf images. It's trained on a dataset featuring different plant diseases**, 
    ensuring accurate and reliable predictions.
    """
)

st.info(
    """
    ⚠️ **Note:** Please upload clear leaf images of **Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato**.
    Images of other plants may not yield accurate results.
    """
)

# Load the pre-trained model
model = keras.models.load_model('trained_model.h5')

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
uploaded_file = st.file_uploader("🌿 Upload a leaf image to detect disease:")
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
        st.success(f"✅ **Prediction:** {label_name[np.argmax(predictions)]} (Confidence: {confidence:.2f}%)")
    else:
        st.warning("⚠️ The model is not confident in its prediction. Please try another image.")
else:
    st.image(
        "https://via.placeholder.com/500x300.png?text=Upload+Your+Leaf+Image+Here",
        caption="Upload a leaf image to start detecting diseases!",
        use_column_width=True,
    )
