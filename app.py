# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from skimage.transform import resize
# from PIL import Image
# import io
# import os

# # Streamlit app
# st.title('Brain Tumor Detection and Classification')

# # Function to load the model
# @st.cache_resource
# def load_keras_model():
#     # Try to load the model, handling potential errors
#     try:
#         # First, try loading the .keras format
#         if os.path.exists('Bmodel.keras'):
#             model = load_model('Bmodel.keras', compile=False)
#         # If .keras doesn't exist, try .h5 format
#         elif os.path.exists('Bmodel.h5'):
#             model = load_model('Bmodel.h5', compile=False)
#         else:
#             st.error("Model file not found. Please ensure 'Bmodel.keras' or 'Bmodel.h5' is in the same directory as this script.")
#             return None
        
#         # Compile the model
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
#     except Exception as e:
#         st.error(f"Error loading the model: {str(e)}")
#         return None

# # Load the model
# model = load_keras_model()

# # Define the labels
# labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# def preprocess_image(image):
#     # Resize the image to match the input size of your model
#     img_resized = resize(image, (224, 224, 3))
#     # Expand dimensions to create a batch
#     img_expanded = np.expand_dims(img_resized, axis=0)
#     return img_expanded

# def predict(image):
#     if model is None:
#         return "Model not loaded", 0

#     # Preprocess the image
#     processed_image = preprocess_image(image)
    
#     # Make prediction
#     prediction = model.predict(processed_image)
    
#     # Get the predicted class index
#     predicted_class_index = np.argmax(prediction[0])
    
#     # Get the predicted class label
#     predicted_class = labels[predicted_class_index]
    
#     # Get the confidence score
#     confidence = prediction[0][predicted_class_index]
    
#     return predicted_class, confidence

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         # Convert PIL Image to numpy array
#         image_array = np.array(image)
        
#         # Make prediction
#         predicted_class, confidence = predict(image_array)
        
#         # Display results
#         st.write(f"Prediction: {predicted_class}")
#         st.write(f"Confidence: {confidence:.2f}")
        
#         # Display a bar chart of class probabilities
#         if model is not None:
#             probabilities = model.predict(preprocess_image(image_array))[0]
#             st.bar_chart({label: prob for label, prob in zip(labels, probabilities)})
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")

# st.write("Note: This is for to just  mess with brain ")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import cv2
import logging

tf.config.run_functions_eagerly(True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.title('Brain Tumor Detection and Classification with Fast Visualization')

# Function to load the model
@st.cache_resource
def load_keras_model():
    try:
        if os.path.exists('Bmodel.keras'):
            model = load_model('Bmodel.keras', compile=False)
        elif os.path.exists('Bmodel.h5'):
            model = load_model('Bmodel.h5', compile=False)
        else:
            st.error("Model file not found. Please ensure 'Bmodel.keras' or 'Bmodel.h5' is in the same directory as this script.")
            return None
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Error loading the model: {str(e)}")
        st.error(f"Error loading the model: {str(e)}")
        return None

# Load the model
model = load_keras_model()

# Define the labels
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    img_resized = resize(image, (224, 224, 3))
    img_resized = img_resized.astype(np.float32)  # Convert to float32
    img_expanded = np.expand_dims(img_resized, axis=0)
    return img_expanded

def predict(image):
    if model is None:
        return "Model not loaded", 0

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    return predicted_class, confidence

@tf.function
def compute_saliency_map(image, model):
    with tf.GradientTape() as tape:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        tape.watch(image)
        predictions = model(image)
        top_class = tf.argmax(predictions[0])
        top_class_score = predictions[0, top_class]

    gradients = tape.gradient(top_class_score, image)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency_map

def fast_visualization(image, model):
    try:
        # Ensure the image is in the correct format
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Compute saliency map
        saliency_map = compute_saliency_map(processed_image, model)
        
        # Convert saliency_map to a NumPy array if it is a TensorFlow tensor
        if isinstance(saliency_map, tf.Tensor):
            saliency_map = saliency_map.numpy()

        # Normalize saliency map
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        
        # Resize saliency map to match original image size
        saliency_map = cv2.resize(saliency_map[0], (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        output = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Find contours and draw circles around detected tumor areas
        gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_heatmap, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # Get the smallest circle that encloses the contour
            center = (int(x), int(y))
            radius = int(radius)
            output = cv2.circle(output, center, radius, (255, 0, 0), 3)  # Draw the circle on the image
        
        return output
    except Exception as e:
        logger.error(f"Error in fast_visualization: {str(e)}")
        return None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        image_array = np.array(image)
        
        predicted_class, confidence = predict(image_array)
        
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
        
        if model is not None:
            probabilities = model.predict(preprocess_image(image_array))[0]
            st.bar_chart({label: prob for label, prob in zip(labels, probabilities)})

        # Add "View" button to visualize the tumor with a circle
        if st.button("View Tumor Visualization"):
            if predicted_class != 'notumor':
                visualization = fast_visualization(image_array, model)
                if visualization is not None:
                    st.image(visualization, caption='Tumor Visualization with Circle', use_column_width=True)
                    st.write("The heatmap overlay shows regions where the model focuses most for its prediction, with a circle marking the tumor location.")
                else:
                    st.warning("Visualization could not be generated. Please check the logs for more information.")
            else:
                st.write("No tumor detected, so no heatmap or circle is generated.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

st.write("Note: This is for fun just to try to mess with your brain.")
