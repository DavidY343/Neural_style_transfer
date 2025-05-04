
from src.video_style_transfer import video_style_transfer
from src.image_style_transfer import image_style_transfer
from src.video_style_transfer_in_medias_res import video_style_transfer_in_medias_res

def nst_main(config):
    """Main function to parse arguments and run the video style transfer."""
    if config["where"] == "img":
        image_style_transfer(config)
    elif config["where"] == "video":
        video_style_transfer(config)
    elif config["where"] == "video_imr":
        video_style_transfer_in_medias_res(config)
    else:
        raise ValueError("Invalid value for 'where'. Expected 'img', 'video', or 'video_imr'.")
    

def main():
    """Main function to parse arguments and run the video style transfer."""

    config = {
        "content_filepath": "video_content/happy_heidi_cow_short_25fps.mp4",
        "style_filepath": "styles/postimpressionism/VanGogh/starry_night.jpg",
        "output_dir": "video_results/",
        "output_image_size": (512, 910),
        "train_config_path": "config/grid_video_config/video_subtil_content.yaml",
        "output_image_format": "jpg",
        "verbose": True,
        "save_intermediate": False,
        "in_medias_res": "frame-00000103.jpg",
        "where": "video_imr",
    }   
    nst_main(config)

if __name__ == '__main__':
	main()