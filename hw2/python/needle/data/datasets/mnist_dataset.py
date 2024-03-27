from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8, offset=16)
    
        # 读取标签文件
        with gzip.open(label_filename, 'rb') as f:
            label_data = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # 图像数据归一化
        image_data = image_data.astype(np.float32)/255.0
        self.images = image_data.reshape((-1, 28, 28, 1))
        self.labels = label_data
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        return img.reshape(-1, 28*28), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION