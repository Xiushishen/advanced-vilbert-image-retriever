import pickle

pkl_file = '/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/cache/RetrievalFlickr8k_train_30_cleaned.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print(f"Number of entries in {pkl_file}: {len(data)}")
for idx, entry in enumerate(data[:5]):
    print(f"Entry {idx}: {entry}")
