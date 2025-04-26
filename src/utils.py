
import yaml
import os
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.utils import save_image

def load_config(config_path, verbose=False):
	DEFAULT_CONFIG = {
        "num_epochs": 3000,
		"learning_rate": 0.001,
		"alpha": 1,
		"beta": 1,
		"capture_content_features_from": {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'},
		"capture_style_features_from": {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
	}

	if config_path is None:
		if verbose:
			print("No config path provided - using DEFAULT configuration")
		return DEFAULT_CONFIG
	train_config = dict()
	if config_path is not None:
		if verbose:
			print("Loading training configuration file...")

		try:
			with open(config_path, 'r') as f:
				train_config = yaml.safe_load(f)
		except FileNotFoundError:
			print(f"ERROR: could not find such file: '{config_path}'.")
			return
		except yaml.YAMLError:
			print(f"ERROR: fail to load yaml file: '{config_path}'.")
			return

		if verbose:
			print("Training configuration file successfully loaded.")
			print()
		return train_config


def load_image(image_path, device, output_size=None):

    img = Image.open(image_path)
    
    # Default to original image size (height, width)
    if output_size is None:
        output_size = img.size[::-1]  # Convert (width, height) to (height, width)
    
    
    if isinstance(output_size, (list, tuple)):
        if len(output_size) == 1:
            output_size = (output_size[0], output_size[0])
        output_size = tuple(output_size)
    else:
        output_size = (output_size, output_size)
    
    
    # Transformaciones (redimensionar + convertir a tensor + normalizar)
    loader = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return loader(img).unsqueeze(0).to(device)

def parse_layers(value):
	if isinstance(value, set):
		return value
	elif isinstance(value, dict):
		return set(value.keys())
	elif isinstance(value, str):
		return set([item.strip() for item in value.split(',')])
	else:
		raise ValueError(f"Invalid layer specification: {value}")


def denormalize_image(tensor, mean, std, device):
    """Desnormaliza la imagen a los valores originales de rango [0, 1]."""

    mean = torch.tensor(mean).to(device).view(1, -1, 1, 1)
    std = torch.tensor(std).to(device).view(1, -1, 1, 1)
    
    denormalized = tensor * std + mean
    return denormalized.clamp(0, 1)

def save_styled_image(generated, path, device):
    denormalized_image = denormalize_image(generated.squeeze(0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device)

    print(f"Saving image to: {path}")
    save_image(denormalized_image, path)


def extract_image_name(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]