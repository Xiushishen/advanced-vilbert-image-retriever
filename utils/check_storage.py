import torch
print(torch.cuda.get_device_name(0))
print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
