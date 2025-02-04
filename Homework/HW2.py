import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA


def normalize_image(img):
    img = np.clip(img, 0, 255)
    return np.uint8((img - img.min()) / (img.max() - img.min()) * 255) if img.max() > img.min() else img


def svd_decomposition(matrix):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def reconstruct_from_svd(U, S, Vt, num_components):
    S_reduced = np.zeros((U.shape[1], Vt.shape[0]))  # Ensure correct shape
    np.fill_diagonal(S_reduced[:num_components, :num_components], S[:num_components])
    return np.dot(U[:, :num_components], np.dot(S_reduced[:num_components, :num_components], Vt[:num_components, :]))


def svd_compression(image, num_components):
    return normalize_image(reconstruct_from_svd(*svd_decomposition(image), num_components))


def pca_compression(image, num_components):
    pca = PCA(n_components=num_components)
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)
    return normalize_image(reconstructed)


def plot_images(original, svd_images, pca_images, percentages):
    fig, axes = plt.subplots(len(percentages) + 1, 3, figsize=(10, 10))
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("Original Image")

    for i, percent in enumerate(percentages):
        axes[i + 1, 0].imshow(svd_images[i], cmap='gray')
        axes[i + 1, 0].set_title(f"SVD {percent}%")

        axes[i + 1, 1].imshow(pca_images[i], cmap='gray')
        axes[i + 1, 1].set_title(f"PCA {percent}%")

    plt.tight_layout()
    plt.show()


def plot_reconstruction_error(S_svd, S_pca, frobenius_errors_svd, frobenius_errors_pca, percentages):
    plt.figure(figsize=(8, 5))
    plt.plot(percentages, np.cumsum(S_svd[:len(percentages)]) / np.sum(S_svd),
             label='SVD Eigenvalues (Variance Contribution)', color='blue')
    plt.plot(percentages, np.cumsum(S_pca[:len(percentages)]) / np.sum(S_pca),
             label='PCA Eigenvalues (Variance Contribution)', color='green')
    plt.plot(percentages, np.log10(frobenius_errors_svd), label='SVD Frobenius Norm Error (log scale)', color='red')
    plt.plot(percentages, np.log10(frobenius_errors_pca), label='PCA Frobenius Norm Error (log scale)', color='orange')
    plt.xlabel('Number of Eigenvalues/Eigenvectors Used')
    plt.ylabel('Log(Error)/Variance Contribution')
    plt.title('Reconstruction Quality Assessment')
    plt.legend()
    plt.grid()
    plt.show()


# Set dimensions for the matrix (M x P with m <= p)
np.random.seed(42)
m, p = 10, 15  # 6 samples, 3 features
A = np.random.randn(m, p)  # Generate a random matrix

# Perform SVD decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruct the matrix using all singular values
A_reconstructed = U @ np.diag(S) @ Vt

# Compute reconstruction error using Frobenius norm
error = np.linalg.norm(A - A_reconstructed, 'fro')

# Print results
print("Original Matrix A:")
print(A)
print("\nSingular Values:")
print(S)
print("\nReconstruction Error (should be close to zero):", error)
# Load the image
image_path = "../Images/pigskull.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}. Check the path and file existence.")

print("Original Image Shape:", original_image.shape)

# Define component percentages
percentages = [10, 25, 50, 100]
num_pixels = min(original_image.shape[:2])  # Use the smaller dimension

# Apply SVD and PCA
svd_images = [svd_compression(original_image, max(1, num_pixels * p // 100)) for p in percentages]
pca_images = [pca_compression(original_image, max(1, num_pixels * p // 100)) for p in percentages]

# Compute Frobenius norm error
frobenius_errors_svd = [np.linalg.norm(original_image - img) for img in svd_images]
frobenius_errors_pca = [np.linalg.norm(original_image - img) for img in pca_images]

# Compute singular values for PCA and SVD
S_svd = svd_decomposition(original_image)[1]
S_pca = PCA(n_components=num_pixels).fit(original_image).singular_values_

# Plot results
plot_images(original_image, svd_images, pca_images, percentages)
plot_reconstruction_error(S_svd, S_pca, frobenius_errors_svd, frobenius_errors_pca, percentages)
