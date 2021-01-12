import torch
from model.model import Network2
dummy_input = torch.rand(1, 3, 64, 64)

model=Network2(output_maps=2)
onnx_path = "onnx_stage2_eye_1.onnx"
torch.onnx.export(model, dummy_input, onnx_path)
