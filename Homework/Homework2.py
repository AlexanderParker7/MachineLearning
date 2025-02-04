from pylab import imshow, gray, figure
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

A = Image.open(r"C:\Users\Ajpti\PythonProjects\MachineLearning\MachineLearning\Images\hummingbird.jpg")  # open an image
print(np.shape(A))  # check the shape of A, it is a 3D tensor
A = np.mean(A, 2)  # get 2-D array by averaging RGB values
m, n = len(A[:, 0]), len(A[1])
r = m / n  # Aspect ratio of the original image
print(f'm/n={r:.3f}, n={n}, A.shape={A.shape}, A.size={A.size}')
fsize, dpi = 3, 80  # inch, dpi (dots per inch, resolution)
plt.figure(figsize=(fsize, fsize * r), dpi=dpi)
U, S, Vh = np.linalg.svd(A, full_matrices=True)
print(U.shape, S.shape, Vh.shape)

# Recover the image
k = 20  # use first k singular values
S = np.resize(S, (m, 1)) * np.eye(m, n)
Compressed_A = np.dot(U[:, 0:k], np.dot(S[0:k, 0:k], Vh[0:k, :]))

plt.figure(figsize=(fsize, fsize * r), dpi=dpi)
gray()
print(f'Image reconstructed using k={k} singular values')
imshow(Compressed_A, cmap='gray')
plt.show()
