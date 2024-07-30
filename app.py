import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import mse, normalize_mse, transform_ssim, convert_to_native_types, save_rankings, load_rankings
from skimage.metrics import structural_similarity as ssim

password = "8888"
# Ensure the submit folder exists
submit_folder = "submit"
if not os.path.exists(submit_folder):
    os.makedirs(submit_folder)

# Function to clear files in the submit directory
def clear_submit_folder():
    for filename in os.listdir(submit_folder):
        file_path = os.path.join(submit_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Failed to delete {file_path}. Reason: {e}")

# Function to load original image based on selected option
def load_original_image(option):
    original_file_path = f"originals/originals_{option}.jpeg"
    if not os.path.exists(original_file_path):
        st.error("선택한 원본 이미지가 존재하지 않습니다.")
        st.stop()
    original_image = cv2.imread(original_file_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return original_image_rgb, original_image

# Streamlit application
st.title("이미지 유사도 측정")

# Default original image
default_original_file_path = "originals/originals_1.jpeg"  # Default original image path
original_image_rgb = None
original_image = None

if os.path.exists(default_original_file_path):
    original_image = cv2.imread(default_original_file_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    st.image(original_image_rgb, caption="원본 이미지", use_column_width=True)

# Load current rankings
rankings = load_rankings()

# Display the top-ranked image under the original image if rankings exist
if rankings:
    top_ranking = rankings[0]
    if "Filename" in top_ranking:
        top_image_path = os.path.join(submit_folder, top_ranking["Filename"])
        if os.path.exists(top_image_path):
            top_image = cv2.imread(top_image_path)
            top_image_rgb = cv2.cvtColor(top_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            st.image(top_image_rgb, caption=f"1등 점수 이미지: {top_ranking['Name']}", use_column_width=True)

    # Create a DataFrame and display the results as a table
    df = pd.DataFrame(rankings)
    df = df[["Name", "SSIM Score (80%)", "MSE Score (20%)", "Final Score (100%)"]]  # Reorder and remove filename
    st.header("현재 랭킹")
    st.dataframe(df)

st.header("비교할 이미지 업로드")

name = st.text_input("이름을 입력하세요")
file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

if name and file:
    if st.button("제출"):
        # Save the uploaded file to the submit folder
        file_path = os.path.join(submit_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"파일 업로드가 완료되었습니다.")

        if original_image is None:
            st.error("원본 이미지가 설정되지 않았습니다. 관리자에게 문의하세요.")
            st.stop()

        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Read comparison image from the saved file
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (gray_original.shape[1], gray_original.shape[0]))

        # Calculate the maximum possible MSE (for a difference of 255 across all pixels)
        max_mse = (255 ** 2)

        # Compute SSIM and MSE
        (ssim_score, diff) = ssim(gray_original, gray_image, full=True)
        mse_score = mse(gray_original, gray_image)

        # Transform SSIM score to 80-point scale
        ssim_final_score = transform_ssim(ssim_score)

        # Normalize MSE score to 20-point scale
        mse_final_score = normalize_mse(mse_score, max_mse)

        final_score = ssim_final_score + mse_final_score

        # Collect result
        result = {
            "Name": name,
            "Filename": file.name,  # Add filename to the result
            "SSIM Score (80%)": ssim_final_score,
            "MSE Score (20%)": mse_final_score,
            "Final Score (100%)": final_score
        }

        # Load current rankings
        rankings = load_rankings()

        # Append new result to rankings
        rankings.append(result)

        # Sort rankings by final score in descending order
        rankings = sorted(rankings, key=lambda x: x["Final Score (100%)"], reverse=True)

        # Save updated rankings
        save_rankings(rankings)

        # Show the diff image for the uploaded image
        best_diff = (diff * 255).astype("uint8")
        st.header("차이 이미지")
        st.image(best_diff, caption=f"원본과의 차이", use_column_width=True)

# Password-protected section to reset rankings and select original image
if st.checkbox("관리자 모드"):
    admin_password = st.text_input("관리자 비밀번호를 입력하세요", type="password", key="admin_password")
    if admin_password == password:
        st.header("원본 이미지 선택 (관리자용)")
        option = st.selectbox("원본 이미지를 선택하세요", [1, 2, 3, 4, 5, 6])
        if st.button("적용", key="apply_button"):
            original_image_rgb, original_image = load_original_image(option)
            st.image(original_image_rgb, caption=f"선택한 원본 이미지 {option}", use_column_width=True)

        st.header("랭킹 초기화")
        reset_password = st.text_input("비밀번호를 입력하세요", type="password", key="reset_password")
        if st.button("랭킹 초기화", key="reset_button"):
            if reset_password == password:
                clear_submit_folder()  # Clear the submit folder
                save_rankings([])  # Reset the rankings
                st.success("랭킹과 제출된 파일이 초기화되었습니다.")
            else:
                st.error("비밀번호가 틀렸습니다.")
