
import os
import PIL
import torch
from torchvision.utils import save_image
from PIL import Image

from src.train_model import train_nst
import src.utils as utils

def image_style_transfer(config):
	"""Performs image style transfer using the provided configuration."""
	
	# Load configuration parameters else use default values
	content_path = config.get('content_filepath')
	style_path = config.get('style_filepath')
	output_dir = config.get("output_dir", ".")
	output_size = config.get("output_image_size", (512, 512))
	verbose = config.get('verbose', False)
	train_config_path = config.get('train_config_path', "config/img_default.yaml")
	output_img_fmt = config.get('output_image_format', 'jpg')

	try:
		content_img = Image.open(content_path)
	except FileNotFoundError:
		print(f"Error: could not find file: '{content_path}'.")
		return
	try:
		style_img = Image.open(style_path)
	except FileNotFoundError:
		print(f"Error: could not find file: '{style_path}'.")
		return
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	content_tensor = utils.load_image(content_path, device, output_size=output_size)
	output_size = (content_tensor.shape[2], content_tensor.shape[3])
	style_tensor = utils.load_image(style_path, device, output_size=output_size)

	generated_tensor = content_tensor.clone().requires_grad_(True)

	if verbose:
		print("Content, style and Output image successfully initialized.")
		print()

	train_config = utils.load_config(train_config_path, verbose=verbose)
	
	if verbose:
		print("Training...")
	
	content_img_name = utils.extract_image_name(content_path)
	style_img_name = utils.extract_image_name(style_path)

	# train model
	success = train_nst(
	content_tensor,
	style_tensor,
	generated_tensor,
	device,
	train_config,
	output_dir,
	output_img_fmt,
	content_img_name,
	style_img_name,
	verbose=False,
	save_intermediate=False  
	)

	final_name = f'{content_img_name}_with_{style_img_name}_final.{output_img_fmt}'
	if success:
		utils.save_styled_image(generated_tensor, os.path.join(output_dir, final_name), device)
	if verbose:
		print(f"Output image successfully generated as {os.path.join(output_dir, final_name)}.")

	