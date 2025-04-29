import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import math

# Configuración
base_path = "D:/ProjectosVSCode/Neural_style_transfer/try"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
substring_filter = "abs"  # Cambia esto por el substring que necesites

# Obtener todas las carpetas que contengan el substring
folders = [f for f in os.listdir(base_path) 
           if os.path.isdir(os.path.join(base_path, f)) and substring_filter in f.lower()]
total_folders = len(folders)
print(f"Total de carpetas encontradas con '{substring_filter}': {total_folders}")

# Configuración de visualización
max_per_page = 20  # Máximo de imágenes por página
cols = 5          # Columnas por página
rows = math.ceil(max_per_page / cols)

# Función para mostrar un conjunto de imágenes
def show_image_page(folders_subset, page_num):
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Página {page_num + 1} - Carpetas con '{substring_filter}'", fontsize=16, y=1.02)
    
    for i, folder in enumerate(folders_subset):
        folder_path = os.path.join(base_path, folder)
        
        # Buscar la primera imagen en la carpeta
        image_path = None
        for ext in image_extensions:
            images = glob(os.path.join(folder_path, f"*{ext}"))
            if images:
                image_path = images[0]
                break
        
        if image_path:
            try:
                # Leer la imagen
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Mostrar en una subtrama
                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(folder[:15] + "..." if len(folder) > 15 else folder, fontsize=8)
                plt.axis('off')
                
            except Exception as e:
                print(f"Error al procesar {image_path}: {e}")
                plt.subplot(rows, cols, i + 1)
                plt.text(0.5, 0.5, "Error", ha='center', va='center')
                plt.axis('off')
        else:
            print(f"No se encontraron imágenes en {folder}")
            plt.subplot(rows, cols, i + 1)
            plt.text(0.5, 0.5, "No image", ha='center', va='center')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar las imágenes en páginas
for page in range(0, total_folders, max_per_page):
    folders_subset = folders[page:page + max_per_page]
    show_image_page(folders_subset, page // max_per_page)