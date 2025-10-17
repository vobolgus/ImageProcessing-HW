import torch
print(torch.__version__)

# Проверка доступности устройства MPS (для Apple Silicon)
if torch.backends.mps.is_available():
    print("MPS device is available.")
    device = torch.device("mps")
else:
    print("MPS device not found. Using CPU.")
    device = torch.device("cpu")

x = torch.rand(5, 3, device=device)
print(x)