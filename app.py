import streamlit as st
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import pandas as pd


# Function to compute Mean Squared Error (MSE)
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Function to normalize MSE score to a 0-20 scale
def normalize_mse(mse_score, max_mse):
    normalized_score = 20 * (1 - mse_score / max_mse)
    return max(0, normalized_score)  # Ensure the score is not negative


# Streamlit application
st.title("Image Similarity Measurement")

st.header("Upload Original Image")
original_file = st.file_uploader("Upload the original image", type=["png", "jpg", "jpeg"], key="original")

st.header("Upload Images for Comparison")
comparison_files = st.file_uploader("Upload up to 3 images for comparison", type=["png", "jpg", "jpeg"],
                                    accept_multiple_files=True, key="comparison")

if original_file and len(comparison_files) == 3:
    # Read original image
    original_image = cv2.imdecode(np.frombuffer(original_file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Read comparison images
    comparison_images = []
    gray_comparison_images = []
    for file in comparison_files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (gray_original.shape[1], gray_original.shape[0]))
        comparison_images.append(image)
        gray_comparison_images.append(gray_image)

    # Calculate the maximum possible MSE (for a difference of 255 across all pixels)
    max_mse = (255 ** 2)

    # Compute SSIM and MSE
    results = []
    diffs = []
    for i, gray_image in enumerate(gray_comparison_images):
        # SSIM
        (ssim_score, diff) = ssim(gray_original, gray_image, full=True)
        diffs.append(diff)

        # MSE
        mse_score = mse(gray_original, gray_image)

        # Normalize scores
        ssim_final_score = ssim_score * 80
        mse_final_score = normalize_mse(mse_score, max_mse)

        final_score = ssim_final_score + mse_final_score

        # Collect results
        results.append({
            "Sample": f"sample{i + 1}",
            "SSIM Score (80%)": ssim_final_score,
            "MSE Score (20%)": mse_final_score,
            "Final Score (100%)": final_score
        })

    # Create a DataFrame and display the results as a table
    df = pd.DataFrame(results)
    st.header("Similarity Results")
    st.dataframe(df)

    # Show the diff image for the most similar image
    best_match_idx = np.argmax(df["Final Score (100%)"])
    best_diff = (diffs[best_match_idx] * 255).astype("uint8")

    st.header("Difference Image with Most Similar Sample")
    st.image(best_diff, caption=f"Difference with {results[best_match_idx]['Sample']}", use_column_width=True)

elif original_file and len(comparison_files) != 3:
    st.warning("Please upload exactly 3 comparison images.")
