import torch

def check_cuda_availability():
    # Verifica si CUDA está disponible
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("¡CUDA está disponible! Puedes usarlo para acelerar tus cálculos.")
        print(f"Dispositivo CUDA actual: {torch.cuda.current_device()}")
        print(f"Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"Número de dispositivos CUDA disponibles: {torch.cuda.device_count()}")
        
        # Información adicional sobre la versión de CUDA
        print(f"Versión de CUDA: {torch.version.cuda}")
        print(f"Versión de PyTorch: {torch.__version__}")
    else:
        print("CUDA no está disponible en tu sistema.")
        print("Posibles razones:")
        print("- No tienes una GPU NVIDIA compatible con CUDA")
        print("- Los drivers de NVIDIA no están instalados correctamente")
        print("- No tienes instalado PyTorch con soporte para CUDA")
        print("- Estás usando una versión de PyTorch incompatible con tu GPU")

if __name__ == "__main__":
    print("Comprobando disponibilidad de CUDA...")
    check_cuda_availability()