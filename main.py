
from src.video_style_transfer import video_style_transfer
from src.image_style_transfer import image_style_transfer


def nst_main(config):
    """Main function to parse arguments and run the video style transfer."""
    if config["video"] == True:
        video_style_transfer(config)
    else:
        image_style_transfer(config)

def main():
    """Main function to parse arguments and run the video style transfer."""

    config6 = {
        "content_filepath": "img_content/little_dog_jumping.jpg",
        "style_filepath": "styles/udnie.jpg",
        "output_dir": "img_results/",
        "output_image_size": (960, 640),
        "train_config_path": "config/img_abstract.yaml",
        "output_image_format": "jpg",
        "verbose": True,
        "save_intermediate": True,
        "video": False
    }   
    nst_main(config6)

if __name__ == '__main__':
	main()