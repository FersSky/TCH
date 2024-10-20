import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, laplace, gaussian_filter
#from imageio import imread
import matplotlib.patches as patches
import argparse


def smoothen(img, display=False):
    """
    Aplica un filtro Gaussiano a la imagen para suavizarla.
    """
    img = gaussian_filter(img, sigma=1)  # Usamos un filtro Gaussiano optimizado
    if display:
        plt.imshow(img, cmap='gray')
        plt.title("Imagen Suavizada")
        plt.show()
    return img

def edge(img, threshold, display=False):
    """
    Detecta bordes usando Sobel y un Laplaciano de Gaussiano (LoG).
    """
    # Detectamos los bordes con Sobel
    G_x = sobel(img, axis=0)  # Gradientes en el eje x
    G_y = sobel(img, axis=1)  # Gradientes en el eje y
    G = np.hypot(G_x, G_y)    # Magnitud del gradiente

    # Aplicamos umbral para detectar bordes mas fuertes
    G[G < threshold] = 0

    # Detectamos los cruces por cero con Laplaciano
    L = laplace(img)
    if L is None:
        return

    # Detectamos cruces por cero
    zero_crossing = np.zeros(L.shape)
    (M, N) = L.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if L[i, j] < 0 and np.any(L[i-1:i+2, j-1:j+2] > 0):
                zero_crossing[i, j] = 1

    # Combinamos con la magnitud del gradiente
    result = np.logical_and(zero_crossing, G).astype(np.uint8)
    
    if display:
        plt.imshow(result, cmap='gray')
        plt.title("Imagen con Bordes Detectados")
        plt.show()

    return result

def detectCircles(img, threshold, region, radius=None):
    """
    Detecta círculos en la imagen de bordes usando la Transformada de Hough Circular.
    """
    (M, N) = img.shape
    if radius is None:
        R_max = max(M, N)
        R_min = 3
    else:
        R_max, R_min = radius

    R_range = R_max - R_min
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Precomputamos angulos
    theta = np.linspace(0, 2 * np.pi, 360)
    edges = np.argwhere(img)

    for r in range(R_range):
        radius = R_min + r
        circle_blueprint = np.zeros((2 * (radius + 1), 2 * (radius + 1)))
        center = (radius + 1, radius + 1)
        
        # Dibujamos el blueprint del circulo
        for angle in theta:
            x = int(np.round(radius * np.cos(angle)))
            y = int(np.round(radius * np.sin(angle)))
            circle_blueprint[center[0] + x, center[1] + y] = 1
        
        constant = np.count_nonzero(circle_blueprint)
        
        for x, y in edges:
            X = [x - radius + R_max, x + radius + R_max]
            Y = [y - radius + R_max, y + radius + R_max]
            A[radius, X[0]:X[1], Y[0]:Y[1]] += circle_blueprint
        
        A[radius][A[radius] < threshold * constant / radius] = 0

    B = np.zeros_like(A)
    for r, x, y in np.argwhere(A):
        region_slice = A[r-region:r+region, x-region:x+region, y-region:y+region]
        p, a, b = np.unravel_index(np.argmax(region_slice), region_slice.shape)
        B[r+(p-region), x+(a-region), y+(b-region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]

def displayCircles(A, file_path):
    """
    Muestra la imagen original con los círculos detectados superpuestos.
    """
    #img = imread(file_path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')

    circle_coords = np.argwhere(A)
    for r, x, y in circle_coords:
        circle = patches.Circle((y, x), r, fill=False, edgecolor='r')
        ax.add_patch(circle)

    #plt.show()
    # Guardar la imagen con los círculos marcados
    cv2.imwrite("Detec_Canny.jpg", img)

    # Imprimir la cantidad de círculos detectados
    cantidad_circulos = len(circle_coords) if circle_coords is not None else 0
    print(f"Se detectaron {cantidad_circulos} círculos.")

    print(f"Imagen guardada como {args.salida_path}")


if __name__ == "__main__":
    # Argumentos para recibir la ruta de la imagen desde la terminal
    parser = argparse.ArgumentParser(description="Detección de círculos con la Transformada de Hough Circular.")
    parser.add_argument("imagen_path", help="Ruta de la imagen a procesar.")
    parser.add_argument("salida_path", type=str, default="./salida.jpg", help="Ruta donde guardar la imagen procesada.")

    args = parser.parse_args()
    img = cv2.imread(args.imagen_path, cv2.IMREAD_GRAYSCALE)

    # Uso del código
    file_path = args.imagen_path
    #img = imread(file_path, as_gray=True)  # Cargamos la imagen en escala de grises

    # Paso 1: Suavizado
    res = smoothen(img, display=False)

    # Paso 2: Detección de bordes
    res = edge(res, threshold=128, display=False)

    # Paso 3: Detección de círculos
    res = detectCircles(res, threshold=8.1, region=15, radius=[30, 1])

    # Paso 4: Mostrar los círculos detectados
    displayCircles(res, file_path)

    

