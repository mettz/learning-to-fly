import sys
import torch
import torch.nn as nn
import h5py

if len(sys.argv) != 2:
    print("Usage: python export_onnx.py <path_to_h5_file>")
    sys.exit(1)

h5_path = sys.argv[1]

# Load weights
with h5py.File(h5_path, "r") as f:
    w0 = torch.tensor(f["actor/layers/0/weights/parameters"][:])
    b0 = torch.tensor(f["actor/layers/0/biases/parameters"][:]).squeeze(0)
    w1 = torch.tensor(f["actor/layers/1/weights/parameters"][:])
    b1 = torch.tensor(f["actor/layers/1/biases/parameters"][:]).squeeze(0)
    w2 = torch.tensor(f["actor/layers/2/weights/parameters"][:])
    b2 = torch.tensor(f["actor/layers/2/biases/parameters"][:]).squeeze(0)

# Define model
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(18, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 4)
        self.tanh = nn.Tanh()

        self.l1.weight.data.copy_(w0)
        self.l1.bias.data.copy_(b0)
        self.l2.weight.data.copy_(w1)
        self.l2.bias.data.copy_(b1)
        self.l3.weight.data.copy_(w2)
        self.l3.bias.data.copy_(b2)

    def forward(self, x):
        x = self.tanh(self.l1(x))
        x = self.tanh(self.l2(x))
        x = self.tanh(self.l3(x))
        return x

# Export to ONNX
model = Actor()
dummy_input = torch.randn(1, 18)
dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}}
onnx_filename = "actor.onnx"
torch.onnx.export(model, dummy_input, onnx_filename, input_names=["obs"], output_names=["actions"], dynamic_axes=dynamic_axes, opset_version=11)
print(f"Exported ONNX model to {onnx_filename}")