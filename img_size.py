import os
from PIL import Image

def get_image_dimensions(folder_path):
    """
    Obtiene las dimensiones de todas las imágenes en el folder especificado.
    
    Args:
        folder_path (str): Ruta al directorio que contiene las imágenes.
    
    Returns:
        dict: Diccionario con nombres de archivo como clave y tuplas (ancho, alto) como valor.
    """
    image_dimensions = {}
    supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            try:
                filepath = os.path.join(folder_path, filename)
                with Image.open(filepath) as img:
                    image_dimensions[filename] = img.size  # (width, height)
            except Exception as e:
                print(f"Error al procesar {filename}: {str(e)}")
    
    return image_dimensions

def print_image_dimensions(image_dimensions):
    """
    Imprime las dimensiones de las imágenes de forma ordenada.
    
    Args:
        image_dimensions (dict): Diccionario con las dimensiones de las imágenes.
    """
    if not image_dimensions:
        print("No se encontraron imágenes en el directorio.")
        return
    
    print("\nDimensiones de las imágenes:")
    print("-" * 40)
    print("{:<30} {:<15}".format("Nombre del archivo", "Dimensiones (ancho x alto)"))
    print("-" * 40)
    
    for filename, dimensions in image_dimensions.items():
        print("{:<30} {:<15}".format(filename, f"{dimensions[0]} x {dimensions[1]}"))

if __name__ == "__main__":
    folder_path = "data/styles"
    dimensions = get_image_dimensions(folder_path)
    print_image_dimensions(dimensions)