import numpy as np
import matplotlib.pyplot as plt

def generate_random_circles(num_circles, img_size=(1200, 1200), filename="image.png"):
    """
    Genera y guarda una imagen con círculos esparcidos aleatoriamente.

    :param num_circles: Número de círculos a generar.
    :param img_size: Tamaño de la imagen (alto, ancho).
    :param filename: Nombre del archivo donde se guardará la imagen.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(0, img_size[1])

    for _ in range(num_circles):
        radius = np.random.randint(20, 50)  # Radio aleatorio entre 5 y 20
        x = np.random.randint(radius, img_size[0] - radius)  # Coordenada x
        y = np.random.randint(radius, img_size[1] - radius)  # Coordenada y

        #circle = plt.Circle((x, y), radius, color=np.random.rand(3,), fill=False)
        circle = plt.Circle((x, y), radius, color=(0,1,0), fill=True)
        ax.add_artist(circle)

    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invertir el eje y para que coincida con las coordenadas de imagen

    plt.savefig(filename)  # Guardar la imagen
    plt.close(fig)  # Cerrar la figura para liberar memoria

# Generar y guardar 5 imágenes con círculos esparcidos
for i in range(5):
    filename = f"./Circulos/circles_{i+1}.png"
    generate_random_circles(num_circles=10, img_size=(1200, 1200), filename=filename)
    print(f"Imagen guardada como {filename}")
