
from src.video_style_transfer import video_style_transfer
from src.image_style_transfer import image_style_transfer

def main():
	"""Main function to parse arguments and run the video style transfer."""

	# config = {
    #     "content_filepath": "img/content/green_bridge.jpg",
    #     "style_filepath": "img/styles/postimpressionism/VanGogh/starry_night_over_the_rhone.jpg",
    #     "output_dir": "img/output/",
    #     "output_image_size": (512, 512),
    #     "train_config_path": "config/img_style_heavy.yaml",
	# 	"output_image_format": "jpg",
    #     "verbose": True,
	# 	"video": True
    # }
	# config = {
    #     "content_filepath": "video_content/happy_heidi_cow_25fps.mp4",
    #     "style_filepath": "styles/postimpressionism/VanGogh/starry_night_over_the_rhone.jpg",
    #     "output_dir": "videos",
    #     "output_image_size": (512, 512),
    #     "train_config_path": "config/video_default.yaml",
	# 	"output_image_format": "jpg",
    #     "verbose": True,
	# 	"video": True
    # }
	config = {
        "content_filepath": "videos/content_frames/frame-00000001.jpg",
        #"style_filepath": "styles/postimpressionism/VanGogh/wheat_field_with_cypresses.jpg",
		"style_filepath": "styles/blue_wave_flow.jpg",
        "output_dir": "videos/",
        "output_image_size": (512, 512),
        "train_config_path": "config/video_mine.yaml",
		"output_image_format": "jpg",
        "verbose": True,
		"video": False
    }
	if config["video"] == True:
		video_style_transfer(config)
	else:
		image_style_transfer(config)	

if __name__ == '__main__':
	main()