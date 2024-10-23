import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, laplace, gaussian_filter
#from imageio import imread
import matplotlib.patches as patches
import argparse




if __name__ == "__main__":
    # Argumentos para recibir la ruta de la imagen desde la terminal
    parser = argparse.ArgumentParser(description="Detección de círculos con la Transformada de Hough Circular.")
    parser.add_argument("imagen_path", help="Ruta de la imagen a procesar.")
    parser.add_argument("salida_path", type=str, default="./salida.jpg", help="Ruta donde guardar la imagen procesada.")

    args = parser.parse_args()
    img = cv2.imread(args.imagen_path, cv2.IMREAD_GRAYSCALE)

    # Uso del código
    file_path = args.imagen_path


    # Verificar que se haya cargado correctamente la imagen
    if img is None:
        print(f"Error: No se pudo cargar la imagen desde {file_path}")
        
    # Make copy of original image
    cimg2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Find contours
    contour,_ = cv2.findContours(255-img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw all detected contours on image in green with a thickness of 1 pixel
    cv2.drawContours(img, contour, -1, color=(0,255,0), thickness=1)

    # Filtrar contornos por circularidad
    for i, cnt in enumerate(contour):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0:
            continue  # Evitar división por cero

        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        
        print ("Area: ", area)
        # Mostrar solo los contornos con circularidad cercana a 1
        if 0.85 < circularity < 1.1:
            print(f"Contorno {i} es circular. Circularidad: {circularity}")
            color = (0 ,100 + (i*10) ,100 + (i*3) )
            #cv2.drawContours(img, [cnt], -1, color, 2)
        else:
            cv2.drawContours(img, [cnt], -1, (255,0,0), 2)

    # Paso 4: Mostrar los círculos detectados
    #    displayCircles(res, file_path)

    # Guardar la imagen con los círculos marcados
    cv2.imwrite("Contornos Circulares.jpg", img)
    

