import numpy as np
import yaml
import os

# Function to compute Mean Squared Error (MSE)
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Function to normalize MSE score to a 0-20 scale with increased variance
def normalize_mse(mse_score, max_mse):
    normalized_score = 20 * (1 - np.sqrt(mse_score / max_mse))
    return max(0, normalized_score)  # Ensure the score is not negative

# Transform SSIM to 0-80 scale
def transform_ssim(ssim_score):
    min_ssim = 0.3
    max_ssim = 0.7
    min_score = 0
    max_score = 80
    transformed_score = (ssim_score - min_ssim) * (max_score - min_score) / (max_ssim - min_ssim) + min_score
    return max(min_score, min(max_score, transformed_score))  # Ensure the score is within the range

# Convert numpy data types to native Python data types
def convert_to_native_types(data):
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    return data

# Save rankings to YAML file
def save_rankings(rankings):
    native_rankings = convert_to_native_types(rankings)
    with open("rank.yaml", "w") as file:
        yaml.safe_dump(native_rankings, file)

# Load rankings from YAML file
def load_rankings():
    if os.path.exists("rank.yaml"):
        with open("rank.yaml", "r") as file:
            data = yaml.safe_load(file)
            if data is not None:
                return data
    return []
