import torch
import numpy as np
import matplotlib.pyplot as plt

# Load all checkpoints

start = 5
end = 100
step = 5

assert (end - start) % step == 0, "Invalid range and step combination"

checkpoint_files = {
    i: f"./output/models/finetune/model-history/{i:03d}.pth"
    for i in range(start, end + 1, step)
}

# Already frozen layers
frozen_layer_keys = [
    'module._block1.0.weight',
    'module._block3.0.weight', 'module._block3.2.weight',
]

# Layers to analyze
unfrozen_layer_keys = [
    'module._block1.2.weight', # 1.0 already frozen
    'module._block2.0.weight', 
    'module._block4.0.weight', 'module._block4.2.weight', 'module._block4.4.weight', 
    'module._block5.0.weight', 'module._block5.2.weight', 'module._block5.4.weight', 
    'module._block6.0.weight', 'module._block6.2.weight', 'module._block6.4.weight']

layer_keys = frozen_layer_keys + unfrozen_layer_keys

# Store weights across epochs for each layer
layer_weights = {key: [] for key in layer_keys}

# Load weights and compute statistics
for epoch, file_path in checkpoint_files.items():
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model_state_dict']
    for layer in layer_keys:
        weights = model_state_dict[layer]
        layer_weights[layer].append(weights.clone().detach())

print("\n"*5)

# Calculate weight differences (L2 norm) between consecutive epochs
weight_diffs = {key: [] for key in layer_keys}
for layer in layer_keys:
    for i in range(1, len(layer_weights[layer])):
        diff = torch.norm(layer_weights[layer]
                          [i] - layer_weights[layer][i-1]).item()
        weight_diffs[layer].append(diff)

# Plot weight changes
plt.figure(figsize=(12, 8))
for layer, diffs in weight_diffs.items():
    plt.plot(range(start + step, end + step, step), diffs, label=layer)

# Determine layers with minimal changes (potentially stable)
stable_layers = {layer: np.mean(diffs)
                 for layer, diffs in weight_diffs.items()}
stable_layers_sorted = sorted(stable_layers.items(), key=lambda x: x[1])
print(stable_layers_sorted)

plt.xlabel('Epoch')
plt.ylabel('Weight Change (L2 Norm)')
plt.title('Weight Stability Across Layers')
plt.legend()
plt.grid(True)
plt.show()
