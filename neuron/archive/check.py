import torch
import platform

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    c = a @ b
    torch.mps.synchronize()
    print("Matmul on MPS worked. Result shape:", tuple(c.shape))
    print("Device: MPS (your M5 GPU)")
else:
    print("Device: CPU only. Something is off, tell me.")