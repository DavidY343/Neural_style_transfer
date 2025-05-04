
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


def video_style_transfer_in_medias_res(config):
    """Performs video style transfer using the provided configuration, with ability to resume from middle."""

    content_video_path = config.get('content_filepath')
    style_path = config.get('style_filepath')
    output_dir = config.get("output_dir", ".")
    output_size = config.get("output_image_size", (512, 512))
    verbose = config.get('verbose', False)
    in_medias_res = config.get('in_medias_res', None)
    train_config_path = config.get('train_config_path', "config/video_default.yaml")

    # Crear directorios si no existen
    os.makedirs(os.path.join(output_dir, "content_frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "transferred_frames"), exist_ok=True)

    if verbose:
        print("Loading content video...")

    # Extraer frames solo si no existen ya
    cap = cv2.VideoCapture(content_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if total_frames == 0:
        print(f"Error: could not retrieve frames from: '{content_video_path}'.")
        return

    # Determinar desde qué frame continuar
    start_frame = 0
    if in_medias_res:
        # Extraer el número del frame del nombre de archivo (ej. "frame-00000042.jpg" -> 42)
        try:
            start_frame = int(in_medias_res.split('-')[1].split('.')[0]) - 1  # -1 porque los frames empiezan en 1
            if verbose:
                print(f"Resuming from frame {start_frame + 1}")
        except:
            print(f"Warning: Could not parse frame number from {in_medias_res}. Starting from beginning.")
            start_frame = 0

    if verbose:
        print("Performing image style transfer for remaining frames...")

    # Procesar solo los frames faltantes
    for i in tqdm(range(start_frame, total_frames), desc="Processing frames", disable=not verbose):
        content_frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
        output_frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
        
        # Saltar si el frame de salida ya existe
        if os.path.exists(output_frame_path):
            if verbose:
                print(f"\tSkipping frame {i+1} (already processed)")
            continue
            
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

    sample_frame = next((f for f in os.listdir(os.path.join(output_dir, "transferred_frames")) 
                        if f.startswith("transferred_frame-")), None)
    if not sample_frame:
        print("Error: No processed frames found")
        return

    output_frame_height, output_frame_width, _ = cv2.imread(
        os.path.join(output_dir, "transferred_frames", sample_frame)).shape
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, cv2_fourcc, output_fps, 
                                 (output_frame_width, output_frame_height), True)

    for i in range(total_frames):
        frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        elif verbose:
            print(f"\tWarning: Missing frame {i+1} in final video")

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')