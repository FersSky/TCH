import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_circular_mask(radius):
    """
    Crea una mascara circular de un cierto radio, normalizada para usar como nucleo de convolucion.
    """
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2  # Define los píxeles dentro del círculo
    kernel = np.zeros((diameter, diameter))
    kernel[mask] = 1
    kernel = kernel / kernel.sum()  # Normaliza el kernel para que los valores sumen 1
    return kernel

def circular_convolution(img, radius):
    """
    Aplica una "convolución circular" a la imagen con una máscara circular.
    """
    # Crear la máscara circular
    kernel = create_circular_mask(radius)
    
    # Aplicar la convolución
    result = cv2.filter2D(img, -1, kernel)
    
    return result

# Cargar la imagen en escala de grises
file_path = './res/HoughCircles.jpg'
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Aplicar la convolución circular con un radio dado
radius = 5  # Elige el radio para el kernel circular
result = circular_convolution(img, radius)

# Mostrar la imagen original y la convolucionada
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Imagen con Convolución Circular")
plt.imshow(result, cmap='gray')

plt.show()
