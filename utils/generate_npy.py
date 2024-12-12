import json
import numpy as np

# 输入 JSONL 文件路径
file_path = "/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/flickr8k_train.jsonline"

# 存储图片名称的列表
image_names = []

# 读取 JSONL 文件，逐行提取 'img_path' 字段
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)  # 解析 JSON 数据
        if 'img_path' in data:  # 检查是否存在 'img_path'
            img_name = data['img_path'].split('.')[0]  # 去掉文件扩展名
            image_names.append(img_name)  # 添加到列表

# 输出到 .npy 文件的路径
output_path = "/home/nvidia/cs7643/vilbert-multi-task/data/flickr8k/cache/flickr_test_ids.npy"

# 保存图片名称列表为 .npy 文件
np.save(output_path, image_names)

print(f"Image base names saved to {output_path}")
