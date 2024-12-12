import numpy as np

file_path = "/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/cache/flickr_test_ids.npy"
data = np.load(file_path)
print(data)
print(data.size)