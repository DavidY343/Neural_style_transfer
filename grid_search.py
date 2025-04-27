from src.video_style_transfer import video_style_transfer
from src.image_style_transfer import image_style_transfer
import os

def grid_search():
    """Run style transfer with multiple combinations of styles and training configs."""
    
    # Configuración base común
    base_config = {
        "content_filepath": "img_content/cow_girl.jpg",
        "output_image_size": (512, 512),
        "output_image_format": "jpg",
        "verbose": False,
        "video": False
    }
    
    # Lista de imágenes de estilo a probar
    style_images = [
        ("styles/ben_giles.jpg", "colorful_flowers"),
        ("styles/udnie.jpg", "udnie")
        ("styles/blue_wave_flow.jpg","blue_wave"),
        ("styles/pawel.jpg","red_fire"),
        ("styles/purple_dream.jpg","purple_dream"),
        ("styles/dances-at-the-spring.jpg","dances-at-the-spring"),
        ("styles/postimpressionism/VanGogh/starry_night_over_the_rhone.jpg","starry_night_over_the_rhone"),
        ("styles/postimpressionism/VanGogh/wheat_field_with_cypresses.jpg","wheat_field_with_cypresses"),
        ("styles/postimpressionism/VanGogh/mount_gaussier_with_the_mas_de_saint_paul_1889.jpg","mount_gaussier"),
        ("styles/postimpressionism/VanGogh/starry_night.jpg","starry_night"),
        ("styles/postimpressionism/VanGogh/the_night_cafe.jpg","the_night_cafe"),
    ]
    
    # Lista de configuraciones de entrenamiento a probar
    train_configs = [
        ("config/video_mine.yaml", "mine"),
        ("config/img_style_heavy.yaml", "heavy"),
        ("config/video_default.yaml", "default"),
        # Añade más configuraciones aquí (ruta, nombre_corto)
    ]
    
    # Directorio base para resultados
    base_output_dir = "videos/grid_search_results/"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Ejecutar todas las combinaciones
    for style_path in style_images:
        for config_path, config_name in train_configs:
            print(f"\nProcessing: {config_name} config with {os.path.basename(style_path)}")
            
            # Crear configuración para esta combinación
            current_config = base_config.copy()
            current_config["style_filepath"] = style_path
            current_config["train_config_path"] = config_path
            
            # Generar nombre de directorio de salida
            output_dir_name = f"{style_name}_{config_name}"
            current_config["output_dir"] = os.path.join(base_output_dir, output_dir_name)
            
            # Crear subdirectorio para estos resultados
            os.makedirs(current_config["output_dir"], exist_ok=True)
            
            # Ejecutar transferencia de estilo
            image_style_transfer(current_config)

def main():
    """Main function to run grid search."""
    grid_search()

if __name__ == '__main__':
    main()