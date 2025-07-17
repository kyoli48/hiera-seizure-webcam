import torch
print("=== GPU Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.mm(x, x)
    print("✅ GPU computation successful")
else:
    print("❌ CUDA not available")
