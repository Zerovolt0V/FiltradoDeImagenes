import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import io

debug = False

# Función para cargar una imagen
def load_image(image_path):
    try:
        image = io.imread(image_path)
        print(f"Imagen cargada correctamente: {image_path}")
        return image
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None

# Función para mostrar una imagen
def show_image_alone(image, title):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para mostrar una imagen
def show_image(original_image, kernel, filtered_image, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')

    kernel_image = axes[1].imshow(kernel, cmap='gray', vmin=np.min(kernel), vmax=np.max(kernel))
    axes[1].set_title('Kernel')
    axes[1].axis('off')

    # Agregar bordes cuadriculados en cada cuadrado del kernel
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(j, i, f"{kernel[i, j]:.2f}", ha='center', va='center', color='blue', fontweight='bold')

    plt.gray()
    fig.colorbar(kernel_image, ax=axes[1])

    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title('Imagen Filtrada')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.show()

# Definir los kernels laplacianos disponibles
kernels_laplacian = [
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
]

# Definir los kernels sobel disponibles
kernels_sobel = [
    np.array([[ 1, 2, 1], [0, 0, 0], [-1,-2,-1]]),
    np.array([[-1,-2,-1], [0, 0, 0], [ 1, 2, 1]]),
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
]


def custom_convolve2d(image, kernel):
    """
    Performs 2D convolution on the input image with the given kernel.
    The kernel is not flipped, following the standard image processing convention.
    """
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output_height = image.shape[0]
    output_width = image.shape[1]
    output_image = np.zeros_like(image, dtype=float)  # Cambiar el tipo de datos a float

    for y in range(output_height):
        for x in range(output_width):
            output_image[y, x] = np.sum(padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1]] * kernel)

    return output_image

def apply_filter(matrix, kernel):
    temp_matrix = custom_convolve2d(matrix, kernel)

    if debug:
        print("\nCálculo de g(x, y) para el filtro:")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                print(f"g({x}, {y}) = {temp_matrix[y, x]}")
        
        print("\nResultado del Filtrado:")
        print(np.round(temp_matrix, decimals=0))
    
    return temp_matrix

"""
def rescale_to_8bit(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    print(f"\nMínimo valor: {min_val}, Máximo valor: {max_val}")

    # Reescalar los valores al rango [0, 7]
    matrix_rescaled = np.round((matrix - min_val) / (max_val - min_val) * 7, 3)

    print("\nTabla de valores reemplazados:")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            print(f"T({matrix[y, x]}) = {matrix_rescaled[y, x]}")

    print("\nResultado del reescalado a rango [0, 7]:")
    print(np.round(matrix_rescaled))
    return matrix_rescaled
"""

def rescale_to_8bit(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)


    if debug:    
        print(f"\nMínimo valor: {min_val}, Máximo valor: {max_val}")
        
        print("Cálculo de m y b:")
        print(f"m = (7 - 0) / ({max_val} - ({min_val}))")
        print(f"b = 0 - m * {min_val}")
    
    m = (7 - 0) / (max_val - min_val)
    b = 0 - m * min_val

    if debug:    
        print(f"m = {m}, b = {b}")
        
        print("\nFunción T(r) = mr + b:")
        print("T(r) = ", m, "r + ", b)
    
    T_r = np.round(m * matrix + b, 4)

    if debug:
    
        print("\nTabla de valores reemplazados:")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                print(f"T({matrix[y, x]}) = {T_r[y, x]}")
    
    print("\nResultado del reescalado a 8 bits:")
    temp_rescaled = np.round(T_r, decimals=0)
    print(temp_rescaled)
    
    return temp_rescaled


# Generar triangulo de pascal para despues calcular la matriz gaussiana aproximada
def generate_pascal_triangle(size):
    triangle = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i+1):
            if j == 0 or j == i:
                triangle[i][j] = 1
            else:
                triangle[i][j] = triangle[i-1][j-1] + triangle[i-1][j]
    return triangle

# Generar la matriz gaussiana aproximada
def generate_gaussian_kernel(size):
    pascal_triangle = generate_pascal_triangle(size)
    filter_vector = pascal_triangle[-1]  # Última fila del triángulo de Pascal
    filter_matrix = np.outer(filter_vector, filter_vector) # Multiplicar el vector por su transpuesta
    sum_filter = np.sum(filter_matrix)
    gaussian_kernel = filter_matrix / sum_filter
    return gaussian_kernel

# Filtro de la media
def mean_filter(matrix, kernel_type, size):
    print("Matriz original:")
    print(matrix)

    if kernel_type == 1:  # Kernel de media simple
        kernel = np.ones((size, size)) / (size * size)
        print("\nKernel de la media simple:")
        print(kernel)
    elif kernel_type == 2:  # Kernel de media gaussiana
        kernel = generate_gaussian_kernel(size)
        print("\nKernel de la media gaussiana:")
        print(kernel)
    else:
        print("Tipo de kernel inválido. Se utilizará el kernel de media simple.")
        kernel = np.ones((size, size)) / (size * size)
        print("\nKernel de la media simple:")
        print(kernel)

    # Aplicar el filtro
    temp_matrix = apply_filter(matrix, kernel)

    # Asegurar que la matriz resultante tenga el tipo de dato correcto
    temp_matrix = temp_matrix.astype(float)

    # Mostrar imagen filtrada
    show_image(matrix, kernel, temp_matrix, "Imagen Filtrada (Media)")
    
    return temp_matrix

# Filtro de la mediana
def median_filter_operation(matrix, size):
    print("Matriz original:")
    print(matrix)
    print(f"\nFiltro de la mediana con tamaño {size}x{size}")
    print("\nCálculo de g(x, y) para el filtro de la mediana:")

    # Encontrar el valor mínimo y máximo de la matriz original
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Reescalar la matriz al rango [0, 255]
    scaled_matrix = (matrix - min_val) / (max_val - min_val) * 255

    print("\nCálculo de g(x, y) para el filtro de la mediana:")

    temp_matrix = np.zeros_like(scaled_matrix)
    pad_width = size // 2
    padded = np.pad(scaled_matrix, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')

    for i in range(temp_matrix.shape[0]):
        for j in range(temp_matrix.shape[1]):
            window = padded[i:i+size, j:j+size].flatten()
            window_sorted = sorted(window)
            median_index = len(window_sorted) // 2

            if debug:
                print(f"g({j}, {i}):")
                print(f"Valores: {window_sorted}")

            if len(window) % 2 == 0:
                median_value = (window_sorted[median_index] + window_sorted[median_index - 1]) / 2
                if debug:
                    print(f"Mediana (par): {median_value}")
            else:
                median_value = window_sorted[median_index]
                if debug:
                    print(f"Mediana (impar): {median_value}")

            temp_matrix[i, j] = np.floor(median_value + 0.5)
            if debug:
                print(f"Resultado: {temp_matrix[i, j]}")

    # Reescalar la matriz resultante al rango original
    temp_matrix = (temp_matrix / 255) * (max_val - min_val) + min_val

    print("\nResultado del filtrado de la mediana:")
    print(temp_matrix)

    # Asegurar que la matriz resultante tenga el tipo de dato correcto
    temp_matrix = temp_matrix.astype(float)

    kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # Mostrar imagen filtrada
    show_image(matrix, kernel, temp_matrix, "Imagen Filtrada (Mediana)")

    return temp_matrix

# Escoger kernel Laplaciano
def kernel_choice_laplacian(matriz):
    print("\nOpciones de kernel para el filtro laplaciano:")
    for i, kernel in enumerate(kernels_laplacian, start=1):
        print(f"{i}. Kernel {i}:\n{kernel}")

    while True:
        kernel_choice = input(f"Ingrese la opción de kernel (1 a {len(kernels_laplacian)}): ")
        if kernel_choice.isdigit() and 1 <= int(kernel_choice) <= len(kernels_laplacian):
            kernel_choice = int(kernel_choice)
            break
        else:
            print("Opción inválida. Por favor, intente nuevamente.")
    laplacian_filter(matriz, kernel_choice)

# Filtro Laplaciano
def laplacian_filter(matrix, kernel_choice):
    print("Matriz original:")
    print(matrix)

    # Verificar si el kernel_choice es válido
    if kernel_choice < 1 or kernel_choice > len(kernels_laplacian):
        print("Opción de kernel inválida. Se utilizará el kernel por defecto.")
        kernel_choice = 1

    # Seleccionar el kernel según la elección del usuario
    kernel = kernels_laplacian[kernel_choice - 1]

    # Mostrar el kernel seleccionado
    print("\nKernel seleccionado:")
    print(kernel)

    # Aplicar el filtro
    temp_matrix = apply_filter(matrix, kernel)

    # Reescalamiento a 8 bits
    temp_matrix = rescale_to_8bit(temp_matrix)

    # Mostrar imagen filtrada
    show_image(matrix, kernel, temp_matrix, "Imagen Filtrada (Laplaciano)")

    return temp_matrix

# Escoger kernel Sobel
def kernel_choice_sobel(matriz):
    print("\nOpciones de kernel para el filtro sobel:")
    for i, kernel in enumerate(kernels_sobel, start=1):
        print(f"{i}. Kernel {i}:\n{kernel}")

    while True:
        kernel_choice = input(f"Ingrese la opción de kernel (1 a {len(kernels_sobel)}): ")
        if kernel_choice.isdigit() and 1 <= int(kernel_choice) <= len(kernels_sobel):
            kernel_choice = int(kernel_choice)
            break
        else:
            print("Opción inválida. Por favor, intente nuevamente.")
    sobel_filter(matriz, kernel_choice)

# Filtro Sobel
def sobel_filter(matrix, kernel_choice):
    print("Matriz original:")
    print(matrix)

    # Verificar si el kernel_choice es válido
    if kernel_choice < 1 or kernel_choice > len(kernels_sobel):
        print("Opción de kernel inválida. Se utilizará el kernel por defecto.")
        kernel_choice = 1

    # Seleccionar el kernel según la elección del usuario
    kernel = kernels_sobel[kernel_choice - 1]

    # Mostrar el kernel seleccionado
    print("\nKernel seleccionado:")
    print(kernel)

    # Aplicar el filtro
    temp_matrix = apply_filter(matrix, kernel)

    # Reescalamiento a 8 bits
    temp_matrix = rescale_to_8bit(temp_matrix)

    # Mostrar imagen filtrada
    show_image(matrix, kernel, temp_matrix, "Imagen Filtrada (Sobel)")

    return temp_matrix

"""
# Función main
def main():
    ########################EJEMPLOS################################################################

    #matriz = np.array([[3, 3, 6, 6, 0], [3, 3, 1, 0, 1], [6, 5, 5, 1, 1], [2, 2, 2, 3, 7], [5, 4, 0, 2, 1]])
    # Ejemplo del pdf
    #matriz = np.array([[1, 1, 1, 1, 0], [1, 5, 5, 1, 0], [1, 5, 5, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    # Ejemplo del video de laplaciano y sobel del aula virtual:
    matriz = np.array([[3, 0, 5, 0], [3, 0, 1, 0], [4, 0, 0, 0], [7, 0, 7, 0]])

    # Ejemplo para mediana
    matriz2 = np.array([[4, 5, 5, 5, 3], [3, 3, 2, 4, 1], [6, 5, 3, 1, 1], [2, 2, 2, 3, 4], [7, 7, 0, 4, 1]])

    while True:
        print("\nMenú:")
        print("1. Filtro de la Media")
        print("2. Filtro de la Mediana")
        print("3. Filtro Laplaciano")
        print("4. Filtro Sobel")
        print("5. Salir")

        choice = input("Ingrese su elección: ")

        match choice:
            case "1":
                kernel_type = int(input("Ingrese el tipo de kernel de media (1 para simple, 2 para gaussiana): "))
                size = int(input(f"Ingrese el tamaño de la matriz de media (2 a {min(matriz.shape) - 1}): "))
                while size < 2 or size >= min(matriz.shape):
                    print("Tamaño inválido. Intente nuevamente.")
                    size = int(input(f"Ingrese el tamaño de la matriz de media (2 a {min(matriz.shape) - 1}): "))
                mean_filter(matriz, kernel_type, size)
            case "2":
                median_filter_operation(matriz2, 3)
            case "3":
                kernel_choice_laplacian(matriz)
            case "4":
                kernel_choice_sobel(matriz)
            case "5":
                break
            case _:
                print("Opción inválida. Por favor, intente nuevamente.")
"""

           
# Función main
def main():
    # Cargar la imagen
    image_path = input("Ingrese la ruta de la imagen: ")
    original_image = load_image(image_path)

    if original_image is None:
        return

    # Eliminar el canal alfa (transparencia) si la imagen tiene 4 canales
    if original_image.shape[-1] == 4:
        original_image = original_image[..., :3]
        print("La imagen tiene 4 canales. Se eliminó el canal alfa.")

    # Convertir la imagen a escala de grises
    grayscale_image = rgb2gray(original_image)

    # Mostrar la imagen original
    show_image_alone(grayscale_image, "Imagen Original")

    # ... (Resto del código principal)

    while True:
        print("\nMenú:")
        print("1. Filtro de la Media")
        print("2. Filtro de la Mediana")
        print("3. Filtro Laplaciano")
        print("4. Filtro Sobel")
        print("5. Salir")

        choice = input("Ingrese su elección: ")

        match choice:
            case "1":
                kernel_type = int(input("Ingrese el tipo de kernel de media (1 para simple, 2 para gaussiana): "))
                size = int(input(f"Ingrese el tamaño de la matriz de media (2 a {min(grayscale_image.shape) - 1}): "))
                while size < 2 or size >= min(grayscale_image.shape):
                    print("Tamaño inválido. Intente nuevamente.")
                    size = int(input(f"Ingrese el tamaño de la matriz de media (2 a {min(grayscale_image.shape) - 1}): "))
                mean_filter(grayscale_image, kernel_type, size)
            case "2":
                median_filter_operation(grayscale_image, 3)
            case "3":
                kernel_choice_laplacian(grayscale_image)
            case "4":
                kernel_choice_sobel(grayscale_image)
            case "5":
                break
            case _:
                print("Opción inválida. Por favor, intente nuevamente.")



if __name__ == "__main__":
    main()