#pip install streamlit-cropper
#We use PIL library - PIL stands for Python Imaging Library
#used to open, manipulate, process, and save images.

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from streamlit_cropper import st_cropper

st.title("Face Mask Detection Project")
model = tf.keras.models.load_model("face-mask-detector.keras")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

camera_image = st.camera_input("Take a picture")

if camera_image:
    image = Image.open(camera_image)

    st.write("Crop the image")

    cropped_img = st_cropper(
        image,
        realtime_update=True,
        box_color='red',
        aspect_ratio=(1, 1)    # Free selection
    )

    st.image(cropped_img, caption="Cropped Image")

    # Resize for CNN
    cropped_img = cropped_img.resize((150, 150))

if st.button("Detect"):
    my_image = None
    if uploaded_file is not None:
        my_image = uploaded_file
    elif cropped_img is not None:
        my_image = cropped_img
    
    if my_image is not None:
        # Display uploaded image
        st.image(my_image, caption="Uploaded Image", width=300)
        if uploaded_file is not None:
            img = Image.open(my_image)
        else:
            img = my_image
        img = img.convert("RGB")
        img = img.resize((150, 150))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result = model.predict(img_array)
        if result[0,0] < 0.51:
            st.success("Person is with mask")
        else:
            st.error("Person is without mask")

    # Perform face mask detection
    else:
        st.error("Please upload or capture an image")