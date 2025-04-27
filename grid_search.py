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
        ("styles/udnie.jpg", "udnie"),
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
    # Video standard configurations
    ("config/video_standard_content.yaml", "vid_std_content"),
    ("config/video_standar_style.yaml", "vid_std_style"),
    ("config/video_standard_content_fast.yaml", "vid_std_content_fast"),
    ("config/video_standard_style_fast.yaml", "vid_std_style_fast"),
    
    # Abstract style configurations
    ("config/video_abstract_content.yaml", "vid_abs_content"),
    ("config/video_abstract_style.yaml", "vid_abs_style"),
    ("config/video_abstract_content_fast.yaml", "vid_abs_content_fast"),
    ("config/video_abstract_style_fast.yaml", "vid_abs_style_fast"),
    
    # Fine textures configurations
    ("config/video_fine_textures_content.yaml", "vid_text_content"),
    ("config/video_fine_textures_style.yaml", "vid_text_style"),
    ("config/video_fine_textures_content_fast.yaml", "vid_text_content_fast"),
    ("config/video_fine_textures_style_fast.yaml", "vid_text_style_fast"),
    
    # Global structure configurations
    ("config/video_global_estructure_content.yaml", "vid_glob_content"),
    ("config/video_global_estructure_style.yaml", "vid_glob_style"),
    ("config/video_global_estructure_content_fast.yaml", "vid_glob_content_fast"),
    ("config/video_global_estructure_style_fast.yaml", "vid_glob_style_fast"),
    
    # Subtle style configurations
    ("config/video_subtil_content.yaml", "vid_subtle_content"),
    ("config/video_subtil_style.yaml", "vid_subtle_style"),
    ("config/video_subtil_content_fast.yaml", "vid_subtle_content_fast"),
    ("config/video_subtil_style_fast.yaml", "vid_subtle_style_fast"),
    ]
    
    # Directorio base para resultados
    base_output_dir = "videos/grid_search_results/"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Ejecutar todas las combinaciones
    for style_path, style_name in style_images:
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