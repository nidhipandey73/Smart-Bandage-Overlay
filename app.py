# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="Smart Bandage AI",
    page_icon="🩹",
    layout="centered"
)

# Center UI
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    padding-top: 0rem !important;
    margin-top: -30px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# -------------------------------
# Geometry Extraction (MODIFIED for numpy input)
# -------------------------------
def extract_wound_geometry(model, image_np, target_size=(512, 512)):

    resized_img = cv2.resize(image_np, target_size)

    results = model(resized_img)
    result = results[0]

    if result.masks is None:
        return None

    all_geometries = []

    for mask_tensor in result.masks.data:

        mask = mask_tensor.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, target_size)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.reshape(-1, 2)

        # PCA
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Major axis
        major_vector = eigenvectors[:, np.argmax(eigenvalues)]
        projected_major = np.dot(points - mean, major_vector)
        major_length = np.max(projected_major) - np.min(projected_major)

        # Minor axis
        minor_vector = eigenvectors[:, np.argmin(eigenvalues)]
        projected_minor = np.dot(points - mean, minor_vector)
        minor_length = np.max(projected_minor) - np.min(projected_minor)

        angle = np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180

        all_geometries.append({
            "center": mean,
            "angle": angle,
            "major_length": major_length,
            "minor_length": minor_length
        })

    if len(all_geometries) == 0:
        return None

    return {
        "image": resized_img,
        "wounds": all_geometries
    }

# -------------------------------
# Bandage Overlay (MODIFIED to RETURN image)
# -------------------------------
def smart_bandage_overlay(geometry, bandage_path):

    if geometry is None:
        return None

    original = geometry["image"].copy()
    wounds = geometry["wounds"]

    bandage = cv2.imread(bandage_path, cv2.IMREAD_UNCHANGED)

    # Convert BGR → RGB
    if bandage.shape[2] == 3:
        bandage = cv2.cvtColor(bandage, cv2.COLOR_BGR2RGB)
    elif bandage.shape[2] == 4:
        bandage[:, :, :3] = cv2.cvtColor(bandage[:, :, :3], cv2.COLOR_BGR2RGB)
    if bandage is None:
        return None

    for wound in wounds:

        center = wound["center"]
        angle = wound["angle"]
        major_length = wound["major_length"]
        minor_length = wound["minor_length"]

        scale_length = int(major_length * 1.8)
        scale_width  = int(minor_length * 2.5)

        resized_bandage = cv2.resize(bandage, (scale_length, scale_width))

        center_bandage = (scale_length // 2, scale_width // 2)
        rot_mat = cv2.getRotationMatrix2D(center_bandage, angle, 1.0)

        rotated = cv2.warpAffine(
            resized_bandage,
            rot_mat,
            (scale_length, scale_width),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0,0)
        )

        cx, cy = int(center[0]), int(center[1])
        h_b, w_b = rotated.shape[:2]

        x_start = cx - w_b // 2
        y_start = cy - h_b // 2

        for i in range(h_b):
            for j in range(w_b):

                if rotated.shape[2] == 4:
                    alpha = (rotated[i, j, 3] / 255.0) * 0.9

                    if alpha > 0:
                        xi = x_start + j
                        yi = y_start + i

                        if 0 <= xi < 512 and 0 <= yi < 512:
                            original[yi, xi] = (
                                alpha * rotated[i, j, :3] +
                                (1 - alpha) * original[yi, xi]
                            )

    return original

# -------------------------------
# Streamlit UI
# -------------------------------
st.markdown(
    "<h1 style='text-align: center;'>🩹 Smart AI Bandage System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center;'>Automated wound detection & adaptive bandage placement</h4>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📤 Upload", "🩹 Result"])

# -------------------------------
# TAB 1: Upload
# -------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload wound image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Apply Bandage"):
            with st.spinner("Processing..."):

                geometry = extract_wound_geometry(model, image_np)

                if geometry is None:
                    st.warning("No wounds detected.")
                else:
                    output = smart_bandage_overlay(geometry, "bandage.png")

                    # Save result in session
                    st.session_state["output"] = output
                    st.session_state["input"] = image_np

                    st.success("Bandage applied! Check Result tab.")

# -------------------------------
# TAB 2: Result
# -------------------------------
with tab2:
    if "output" in st.session_state:

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                st.session_state["input"],
                caption="Original",
                use_container_width=True
            )

        with col2:
            st.image(
                st.session_state["output"],
                caption="Bandaged",
                use_container_width=True
            )

    else:
        st.info("Upload and process an image first.")
