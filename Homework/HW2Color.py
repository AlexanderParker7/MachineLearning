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
    channels = cv2.split(image)
    compressed_channels = [normalize_image(reconstruct_from_svd(*svd_decomposition(c), num_components)) for c in
                           channels]
    return cv2.merge(compressed_channels)


def pca_compression(image, num_components):
    channels = cv2.split(image)
    compressed_channels = []
    for c in channels:
        pca = PCA(n_components=num_components)
        transformed = pca.fit_transform(c)
        reconstructed = pca.inverse_transform(transformed)
        compressed_channels.append(normalize_image(reconstructed))
    return cv2.merge(compressed_channels)


def plot_images(original, svd_images, pca_images, percentages):
    fig, axes = plt.subplots(len(percentages) + 1, 3, figsize=(10, 10))
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")

    for i, percent in enumerate(percentages):
        axes[i + 1, 0].imshow(cv2.cvtColor(svd_images[i], cv2.COLOR_BGR2RGB))
        axes[i + 1, 0].set_title(f"SVD {percent}%")

        axes[i + 1, 1].imshow(cv2.cvtColor(pca_images[i], cv2.COLOR_BGR2RGB))
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


# Generate a random matrix for proof of SVD recovery
np.random.seed(42)
A = np.random.rand(10, 10)
U, S, Vt = svd_decomposition(A)
A_reconstructed = reconstruct_from_svd(U, S, Vt, 10)

# Verify recovery
error = np.linalg.norm(A - A_reconstructed)
print(f"Reconstruction Error (should be close to zero): {error}")

# Load the image
image_path = "../Images/pigskull.jpg"
original_image = cv2.imread(image_path)

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
S_svd = svd_decomposition(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY))[1]
S_pca = PCA(n_components=num_pixels).fit(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)).singular_values_

# Plot results
plot_images(original_image, svd_images, pca_images, percentages)
plot_reconstruction_error(S_svd, S_pca, frobenius_errors_svd, frobenius_errors_pca, percentages)