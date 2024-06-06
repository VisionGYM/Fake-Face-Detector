from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/Colab Notebooks/img_align_celeba.zip" "."
!unzip "./img_align_celeba.zip" -d "./CelebA/"

!pip install ipython==7.34.0
!pip install --quiet "ipython[notebook]>=8.12.0" "torch>=1.14.0" "setuptools" "torchmetrics>=0.12" "torchvision" "pytorch-lightning>=2.0.0"

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os

# 두 개의 별도 디렉토리에서 이미지를 로드하고 레이블링하는 커스텀 데이터셋

class CelebAGANDataset(Dataset):
    def __init__(self, celeba_dir, gan_dir, transform=None):
        self.celeba_dir = celeba_dir
        self.gan_dir = gan_dir
        self.transform = transform

        self.celeba_images = os.listdir(celeba_dir)
        self.gan_images = os.listdir(gan_dir)

        self.images = self.celeba_images + self.gan_images
        self.labels = [0] * len(self.celeba_images) + [1] * len(self.gan_images)

        # 이미지와 레이블을 같이 섞기
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < len(self.celeba_images):
            img_path = os.path.join(self.celeba_dir, self.images[idx])
        else:
            img_path = os.path.join(self.gan_dir, self.images[idx - len(self.celeba_images)])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label



class CelebAGANDataModule(L.LightningDataModule):
    def __init__(self, celeba_dir: str = "./CelebA", gan_dir: str = "./GAN", batch_size: int = 128):
        super().__init__()
        self.celeba_dir = celeba_dir
        self.gan_dir = gan_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # 다운로드 작업 없음
        pass

    def setup(self, stage=None):
        dataset = CelebAGANDataset(self.celeba_dir, self.gan_dir, transform=self.transform)

        # 데이터셋을 학습용, 검증용, 테스트용으로 분할
        val_size = int(0.1 * len(dataset))
        test_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        # 학습 데이터로더
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # 검증 데이터로더
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # 테스트 데이터로더
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


data_module = CelebADataModule(celeba_dir="./CelebA", gan_dir="./GAN", batch_size=128)
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
