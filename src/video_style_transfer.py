
import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm

from src.train_model import train_nst
import src.utils as utils

def _image_style_transfer(content_frame_path, style_path, output_frame_path, output_size, verbose, config_path):
	try:
		content_img = Image.open(content_frame_path)
	except FileNotFoundError:
		print(f"Error: could not find file: '{content_frame_path}'.")
		return

	try:
		style_img = Image.open(style_path)
	except FileNotFoundError:
		print(f"Error: could not find file: '{style_path}'.")
		return

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	content_tensor = utils.load_image(content_frame_path, device, output_size=output_size)
	output_size = (content_tensor.shape[2], content_tensor.shape[3])
	style_tensor = utils.load_image(style_path, device, output_size=output_size)

	generated_tensor = content_tensor.clone().requires_grad_(True)

	train_config = utils.load_config(config_path, verbose)
	# train model
	success = train_nst(
	content_tensor,
	style_tensor,
	generated_tensor,
	device,
	train_config,
	output_img_fmt='jpg',
	verbose=verbose,
	save_intermediate=False,  # No need to save intermediate images for video processing
	)


	if success:
		utils.save_styled_image(generated_tensor, output_frame_path, device)

	return success

def video_style_transfer(config):
	"""Performs video style transfer using the provided configuration."""

	content_video_path = config.get('content_filepath')
	style_path = config.get('style_filepath')
	output_dir = config.get("output_dir", ".")
	output_size = config.get("output_image_size", (512, 512))
	verbose = config.get('verbose', False)
	train_config_path = config.get('train_config_path', "config/video_default.yaml")


	if not os.path.exists(os.path.join(output_dir, "content_frames")):
		os.makedirs(os.path.join(output_dir, "content_frames"))

	if verbose:
		print("Loading content video...")

	cap = cv2.VideoCapture(content_video_path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	content_fps = cap.get(cv2.CAP_PROP_FPS)
	
	if total_frames == 0:
		print(f"Error: could not retrieve frames from: '{content_video_path}'.")
		return

	# extract frames from content video
	for i in range(total_frames):
		success, img = cap.read()
		if not success:
			print(f"Warning: Failed to read frame {i+1}")
			continue
			
		frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
		cv2.imwrite(frame_path, img)
	cap.release()

	if verbose:
		print("Frames successfully extracted from content video.")
		print()
		print("Performing image style transfer for each frame...")

	if not os.path.exists(os.path.join(output_dir, "transferred_frames")):
		os.makedirs(os.path.join(output_dir, "transferred_frames"))

	
	# Procesar frames con barra de progreso
	for i in tqdm(range(total_frames), desc="Processing frames", disable=not verbose):
		content_frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
		output_frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
		
		try:
			success = _image_style_transfer(content_frame_path, style_path, 
										output_frame_path, output_size, 
										verbose, train_config_path)
			if not success and verbose:
				print(f"\tWarning: Failed to process frame {i+1}")
		except Exception as e:
			if verbose:
				print(f"\tError processing frame {i+1}: {str(e)}")
			continue
	
	if verbose:
		print("Image style transfer complete.")
		print()
		print("Synthesizing video from transferred frames...")
	
	content_video_name = utils.extract_image_name(content_video_path)
	style_img_name = utils.extract_image_name(style_path)
	output_video_path = os.path.join(output_dir, f"nst-{content_video_name}-{style_img_name}-final.mp4")

	output_frame_height, output_frame_width, _ = cv2.imread(os.path.join(output_dir, "transferred_frames", "transferred_frame-00000001.jpg")).shape
	output_fps = config.get('fps') if config.get('fps') is not None else content_fps

	# synthesize video using transferred content frames
	cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video_writer = cv2.VideoWriter(output_video_path, cv2_fourcc, output_fps, (output_frame_width, output_frame_height), True)

	for i in range(total_frames):
		frame = cv2.imread(os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg"))
		if frame is not None:
			video_writer.write(frame)

	video_writer.release()

	if verbose:
		print(f'Video successfully synthesized to {output_video_path}.')
