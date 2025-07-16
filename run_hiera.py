import torch

# Load the Hiera model pretrained on Kinetics 400
model = torch.hub.load("facebookresearch/hiera", model="hiera_base_16x224", pretrained=True, checkpoint="mae_k400_ft_k400")

# Set the model to evaluation mode
model.eval()

# Create a valid input tensor (e.g., 1 video with 16 frames)
# Each frame should have 3 channels (RGB)
batch_size = 1
num_frames = 16
input_tensor = torch.randn(batch_size, num_frames, 3, 224, 224)  # Correct shape

# Run inference
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_tensor)

# Process the output as needed
print(output)
