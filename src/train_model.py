import os
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.utils import save_image
from torchvision import transforms

from src.utils import parse_layers, save_styled_image

valid_layers = {
    'conv1_1', 'conv1_2',
    'conv2_1', 'conv2_2',
    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'
}


class ImageStyleTransfer_VGG19(nn.Module):
	"""
	VGG19 for style transfer. Extracts features from key convolutional layers:

	- conv11 (Block 1): Captures basic textures/edges
	- conv21 (Block 2): Mid-level patterns
	- conv31 (Block 3): Complex textures
	- conv41 (Block 4): Content structure
	- conv51 (Block 5): High-level content

	Uses first 29 layers (conv+ReLU+pooling) excluding FC layers to preserve spatial info.
	Selected layers provide optimal style-content balance.
	"""
	def __init__(self):
		super(ImageStyleTransfer_VGG19, self).__init__()

		self.chosen_features = {
			0: 'conv1_1', 2: 'conv1_2',   # Bloque 1
			5: 'conv2_1', 7: 'conv2_2',   # Bloque 2
			10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3', 16: 'conv3_4',  # Bloque 3
			19: 'conv4_1', 21: 'conv4_2', 23: 'conv4_3', 25: 'conv4_4',  # Bloque 4
			28: 'conv5_1', 30: 'conv5_2', 32: 'conv5_3', 34: 'conv5_4'   # Bloque 5
		}
		self.model = vgg19(weights='DEFAULT').features[:29]

	def forward(self, x):
		feature_maps = dict()
		for idx, layer in enumerate(self.model):
			x = layer(x)
			if idx in self.chosen_features.keys():
				feature_maps[self.chosen_features[idx]] = x
		
		return feature_maps


def _get_content_loss(content_feature, generated_feature):
	"""Compute MSE between content feature map and generated feature map as content loss."""
	return torch.mean((generated_feature - content_feature) ** 2)


def _get_style_loss(style_feature, generated_feature):
	"""Compute MSE between gram matrix of style feature map and of generated feature map as style loss."""
	_, channel, height, width = generated_feature.shape
	style_gram = style_feature.view(channel, height*width).mm(
		style_feature.view(channel, height*width).t()
	)
	generated_gram = generated_feature.view(channel, height*width).mm(
		generated_feature.view(channel, height*width).t()
	)

	return torch.mean((generated_gram - style_gram) ** 2) / (channel * height * width)
	

def check_early_stopping(epochs_no_improve, patience, min_loss, best_loss, total_loss):
	"""Check if early stopping criteria are met."""
	if epochs_no_improve >= patience:
		return True
	if best_loss - total_loss < min_loss:
		epochs_no_improve += 1
	else:
		epochs_no_improve = 0
		best_loss = total_loss

	return False


def train_nst(content, style, generated, device, train_config, output_dir=None,
			output_img_fmt='jpg', content_img_name='content', style_img_name='style',
			verbose=False, save_intermediate=False):
	"""Apply Neural Style Transfer (NST) to generate a stylized image or a frame"""

	model = ImageStyleTransfer_VGG19().to(device).eval()

	# Load configuration parameters else use default values
	num_epochs = train_config.get('num_epochs', 2000)
	lr = train_config.get('learning_rate', 0.01)
	alpha = train_config.get('alpha', 50)
	beta = train_config.get('beta', 0.001)
	capture_content_features_from = train_config.get('capture_content_features_from', 
		{'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'})
	capture_style_features_from = train_config.get('capture_style_features_from', 
		{'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'})
	early_stopping = train_config.get('early_stopping', False)
	patience = train_config.get('patience', 100)
	min_loss = train_config.get('min_loss', 0.01)
	try:
		capture_content_features_from = parse_layers(capture_content_features_from)
		capture_style_features_from = parse_layers(capture_style_features_from)
	except ValueError as e:
		print(f"Error: {e}")
		return 0

	if not capture_content_features_from.issubset(valid_layers):
		print(f"ERROR: invalid content layers: {capture_content_features_from}")
		return 0
	if not capture_style_features_from.issubset(valid_layers):
		print(f"ERROR: invalid style layers: {capture_style_features_from}")
		return 0

	optimizer = torch.optim.Adam([generated], lr=lr)

	# Directorio para guardar intermedios (si aplica)
	if save_intermediate and verbose and output_dir:
		intermediate_dir = os.path.join(output_dir, f'{content_img_name}_with_{style_img_name}')
		os.makedirs(intermediate_dir, exist_ok=True)
	
	content_features = model(content)
	style_features = model(style)

	#early stopping variables
	best_loss = float('inf')
	epochs_no_improve = 0

	# Entrenamiento
	for epoch in range(num_epochs):
		
	
		generated_features = model(generated)

		content_loss = style_loss = 0

		for layer_name in generated_features.keys():
			if layer_name in capture_content_features_from:
				content_loss += _get_content_loss(content_features[layer_name].detach(), generated_features[layer_name])
			if layer_name in capture_style_features_from:
				style_loss += _get_style_loss(style_features[layer_name].detach(), generated_features[layer_name])

		total_loss = alpha * content_loss + beta * style_loss
		
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		if verbose and save_intermediate and output_dir and (epoch + 1) % 200 == 0:
			print(f"Content loss: {alpha * content_loss.item()}, Style loss: {beta * style_loss.item()}")
			path = os.path.join(intermediate_dir, f'{content_img_name}_with_{style_img_name}_{epoch + 1}.{output_img_fmt}')
			save_styled_image(generated, path, device)
			print(f"\tEpoch {epoch + 1}/{num_epochs}, loss = {total_loss.item()}")
		

		if early_stopping and check_early_stopping(epochs_no_improve, patience, min_loss, best_loss, total_loss.item()):
			print(f"Early stopping at epoch {epoch + 1}/{num_epochs}, loss = {total_loss.item()}")
			break

	if verbose and save_intermediate and output_dir:
		print(f"\tFinal images saved in: '{intermediate_dir}'")

	return 1