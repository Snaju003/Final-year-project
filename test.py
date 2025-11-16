import torch

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
else:
    print("⚠️ CUDA not available. Using CPU.")
