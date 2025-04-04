import numpy as np
import cv2

def compute_mean_and_cov(image):
    """Compute mean and covariance of the color distribution."""
    reshaped = image.reshape(-1, 3).astype(np.float32)
    mean = np.mean(reshaped, axis=0)
    cov = np.cov(reshaped, rowvar=False)
    return mean, cov

def sqrtm(matrix, method="svd"):
    """Compute the square root of a positive semi-definite matrix."""
    if method == "svd":
        U, S, Vt = np.linalg.svd(matrix)
        return np.dot(U, np.dot(np.diag(np.sqrt(S)), Vt))
    elif method == "eigen":
        eigvals, eigvecs = np.linalg.eigh(matrix)
        sqrt_diag = np.diag(np.sqrt(eigvals))
        return eigvecs @ sqrt_diag @ eigvecs.T
    elif method == "cholesky":
        L = np.linalg.cholesky(matrix)
        return L
    else:
        raise ValueError("Invalid method for matrix square root")

def separable_transfer(target, reference):
    """Separable linear transfer (Reinhard et al.) - Matches means and variances independently."""
    target = target.astype(np.float32) / 255.0
    reference = reference.astype(np.float32) / 255.0

    mu_t, cov_t = compute_mean_and_cov(target)
    mu_r, cov_r = compute_mean_and_cov(reference)

    scale = np.sqrt(np.diag(cov_r)) / np.sqrt(np.diag(cov_t))
    transform = np.diag(scale)

    transformed = np.dot((target.reshape(-1, 3) - mu_t), transform.T) + mu_r
    transformed = np.clip(transformed, 0, 1)
    
    return (transformed.reshape(target.shape) * 255).astype(np.uint8)

def cholesky_transfer(target, reference):
    """Cholesky-based transfer."""
    target = target.astype(np.float32) / 255.0
    reference = reference.astype(np.float32) / 255.0

    mu_t, cov_t = compute_mean_and_cov(target)
    mu_r, cov_r = compute_mean_and_cov(reference)

    L_t = np.linalg.cholesky(cov_t)
    L_r = np.linalg.cholesky(cov_r)

    transform = L_r @ np.linalg.inv(L_t)
    
    transformed = np.dot((target.reshape(-1, 3) - mu_t), transform.T) + mu_r
    transformed = np.clip(transformed, 0, 1)

    return (transformed.reshape(target.shape) * 255).astype(np.uint8)

def pca_transfer(target, reference):
    """Principal Component Analysis (PCA)-based transfer."""
    target = target.astype(np.float32) / 255.0
    reference = reference.astype(np.float32) / 255.0

    mu_t, cov_t = compute_mean_and_cov(target)
    mu_r, cov_r = compute_mean_and_cov(reference)

    sqrt_cov_t = sqrtm(cov_t, method="eigen")
    sqrt_cov_r = sqrtm(cov_r, method="eigen")

    transform = sqrt_cov_r @ np.linalg.inv(sqrt_cov_t)

    transformed = np.dot((target.reshape(-1, 3) - mu_t), transform.T) + mu_r
    transformed = np.clip(transformed, 0, 1)

    return (transformed.reshape(target.shape) * 255).astype(np.uint8)

def monge_kantorovitch_transfer(target, reference):
    """Monge-Kantorovitch Optimal Transport-based transfer."""
    target = target.astype(np.float32) / 255.0
    reference = reference.astype(np.float32) / 255.0

    mu_t, cov_t = compute_mean_and_cov(target)
    mu_r, cov_r = compute_mean_and_cov(reference)

    sqrt_cov_t = sqrtm(cov_t, method="svd")
    inv_sqrt_cov_t = np.linalg.inv(sqrt_cov_t)
    mk_transform = inv_sqrt_cov_t @ sqrtm(sqrt_cov_t @ cov_r @ sqrt_cov_t, method="svd") @ inv_sqrt_cov_t

    transformed = np.dot((target.reshape(-1, 3) - mu_t), mk_transform.T) + mu_r
    transformed = np.clip(transformed, 0, 1)

    return (transformed.reshape(target.shape) * 255).astype(np.uint8)

def convert_color_space(image, from_space, to_space):
    """Convert image between different color spaces."""
    conversion_code = {
        ("RGB", "YUV"): cv2.COLOR_RGB2YUV,
        ("YUV", "RGB"): cv2.COLOR_YUV2RGB,
        ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
        ("LAB", "RGB"): cv2.COLOR_LAB2RGB,
        ("RGB", "XYZ"): cv2.COLOR_RGB2XYZ,
        ("XYZ", "RGB"): cv2.COLOR_XYZ2RGB,
        ("RGB", "LUV"): cv2.COLOR_RGB2LUV,
        ("LUV", "RGB"): cv2.COLOR_LUV2RGB
    }
    return cv2.cvtColor(image, conversion_code[(from_space, to_space)])

import matplotlib.pyplot as plt

def run_all_transfers(target_path, reference_path, color_space="RGB"):
    """Run all color transfer methods and display results using Matplotlib."""
    target = cv2.imread(target_path)
    reference = cv2.imread(reference_path)

    if color_space != "RGB":
        target = convert_color_space(target, "RGB", color_space)
        reference = convert_color_space(reference, "RGB", color_space)

    results = {
        "Target": target,
        "Reference": reference,
        "Separable": separable_transfer(target, reference),
        "Cholesky": cholesky_transfer(target, reference),
        "PCA": pca_transfer(target, reference),
        "Monge-Kantorovitch": monge_kantorovitch_transfer(target, reference)
    }

    if color_space != "RGB":
        for key in results:
            results[key] = convert_color_space(results[key], color_space, "RGB")

    # Convert images from BGR (OpenCV default) to RGB for matplotlib
    for key in results:
        results[key] = cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB)

    # Plot results
    plt.figure(figsize=(12, 8))
    for i, (name, img) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


# Example Usage
if __name__ == "__main__":
    target_img_path = "/home/neelraj-reddy/college/6th_sem/computer vision/project/A little survey on previous works/images/reference.jpeg"
    reference_img_path = "/home/neelraj-reddy/college/6th_sem/computer vision/project/A little survey on previous works/images/input1.jpg"

    # Run all transfers in RGB
    run_all_transfers(target_img_path, reference_img_path, "RGB")

    # Run all transfers in LAB
    run_all_transfers(target_img_path, reference_img_path, "LAB")

    # Run all transfers in YUV
    run_all_transfers(target_img_path, reference_img_path, "YUV")

    # Run all transfers in CIELUV
    run_all_transfers(target_img_path, reference_img_path, "LUV")
