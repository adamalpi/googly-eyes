import math

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image


INFERENCE_URI = "http://backend:9090/find_eyes/"


def find_eyes(image: np.ndarray) -> str:
    """Call backend service to find the eyes in the image."""
    image.seek(0)
    url = INFERENCE_URI

    files = {"file": image}
    result = requests.post(url, files=files)

    pred_result = result.json()
    return pred_result["eyes"]


def add_googly_eyes(image: np.ndarray, eyes: list):
    """Replace human eyes by a carefully designed googly eyes.

    The algorithm can certainly be improved to make it prettier
    but it won't get funnier 😜
    """
    for center, diameter in eyes:
        diameter = round(diameter)
        cv2.circle(image, center, diameter, color=(0,0,0), thickness=2)
        cv2.circle(image, center, diameter-1, color=(255,255,255), thickness=-1)

        # paint the pupil by randomly selected the angel it will be looking at
        radius = round(diameter*0.5)
        random_angle = np.random.randint(360)
        random_x = center[0] + round(radius * math.cos(random_angle))
        random_y = center[1] + round(radius * math.sin(random_angle))

        cv2.circle(image, (random_x, random_y), round(diameter*0.5), color=(0,0,0), thickness=-1)

    return image


if __name__ == "__main__":
    st.title("Googly Eyes 👀")

    uploaded_image = st.file_uploader("Image to classify.")
    if uploaded_image is not None:
        eyes = find_eyes(uploaded_image)

        image = Image.open(uploaded_image)
        img_array = np.array(image)
        image = add_googly_eyes(img_array, eyes)

        st.image(image)
