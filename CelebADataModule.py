from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/Colab Notebooks/img_align_celeba.zip" "."
!unzip "./img_align_celeba.zip" -d "./CelebA/"

!pip install ipython==7.34.0
!pip install --quiet "ipython[notebook]>=8.12.0" "torch>=1.14.0" "setuptools" "torchmetrics>=0.12" "torchvision" "pytorch-lightning>=2.0.0"

import pytorch_lightning as L
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split

class CelebADataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "./CelebA", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # download 작업 없음
        pass

    def setup(self, stage=None):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)

        # 검증용과 테스트용으로 분할
        val_size = int(0.1 * len(dataset))
        test_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size - test_size
        self.dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])


    def train_dataloader(self):
        # 학습 데이터로더
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # 검증 데이터로더
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # 테스트 데이터로더
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

#사용예시
data_module = CelebADataModule(data_dir="./CelebA", batch_size=128)
data_module.setup()
train_loader = data_module.train_dataloader()

# train_loader 확인용코드
import matplotlib.pyplot as plt
import numpy as np
for batch in train_loader:
    images, labels= batch

    for i in range(len(images)):
        # 정규화 역변환 : [-1, 1] 범위 -> [0, 1] 범위
        image = (images[i] * 0.5) + 0.5

        plt.subplot(8, 16, i+1)
        plt.imshow(image.permute(1, 2, 0))  # 이미지의 차원 순서를 변경하여 출력
        plt.axis('off')
    plt.gcf().set_size_inches(20, 10)
    # 첫 번째 배치만 확인
    break
