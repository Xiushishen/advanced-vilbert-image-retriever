import numpy as np
import random

file_path = "/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/cache/flickr_test_ids1.npy"
data = np.load(file_path)

random.seed(42)
indices_to_keep = random.sample(range(data.size), int(2 * data.size / 3))
indices_to_keep.sort()
reduced_data = data[indices_to_keep]

output_path = "/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/cache/flickr_test_ids.npy"
np.save(output_path, reduced_data)

print("Original size:", data.size)
print("Reduced size:", reduced_data.size)
print("Saved reduced data to:", output_path)
