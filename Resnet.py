import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.functional import accuracy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Tensor Cores 활용을 위한 설정
torch.set_float32_matmul_precision('high')  # 또는 'medium'

# 기본 블록 정의
class BasicBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """_summary_

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
        """        
        super(BasicBlock, self).__init__()
        # 합성곱층의 정의
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        # 다운샘플링 레이어
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 배치 정규화층 정의
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 스킵 커넥션을 위해 초기 입력 저장
        x_ = x
        # ResNet 기본 블록에서 F(x) 부분
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        # 다운샘플링된 원본 입력과 합산
        x_ = self.downsample(x_)

        #합성곱층의 결과와 저장해놨던 입력값을 더함(스킵 커넥션)
        x += x_
        x = self.relu(x)
        return x

# ResNet 모델 정의
class ResNet(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """    
    def __init__(self, num_classes=10, lr=1e-4):
        """_summary_

        Args:
            num_classes (int, optional): _description_. Defaults to 10.
            lr (_type_, optional): _description_. Defaults to 1e-4.
        """        
        super(ResNet, self).__init__()
        # 학습 주기
        self.lr = lr
        # 기본 블록
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)

        # 평균 풀링 레이어
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 첫 번째 완전 연결 레이어
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        # 두 번째 완전 연결 레이어
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        # 세 번째 완전 연결 레이어 (출력 레이어)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

        # ReLU 활성화 함수
        self.relu = nn.ReLU()
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 첫 번째 블록과 풀링
        x = self.b1(x)
        x = self.pool(x)
        # 두 번째 블록과 풀링
        x = self.b2(x)
        x = self.pool(x)
        # 세 번째 블록과 풀링
        x = self.b3(x)
        x = self.pool(x)

        # 평탄화
        x = torch.flatten(x, start_dim=1)

        # 첫 번째 완전 연결 레이어와 ReLU
        x = self.fc1(x)
        x = self.relu(x)
        # 두 번째 완전 연결 레이어와 ReLU
        x = self.fc2(x)
        x = self.relu(x)
        # 출력 레이어
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 학습 단계
        data, label = batch
        preds = self(data)
        loss = self.criterion(preds, label)
        acc = accuracy(preds, label, task='multiclass', num_classes=10)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 검증 단계
        data, label = batch
        preds = self(data)
        loss = self.criterion(preds, label)
        acc = accuracy(preds, label, task='multiclass', num_classes=10)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 테스트 단계
        data, label = batch
        preds = self(data)
        loss = self.criterion(preds, label)
        acc = accuracy(preds, label, task='multiclass', num_classes=10)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        # 옵티마이저 구성
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# CIFAR10 데이터 모듈 정의
class CIFAR10DataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """    
    def __init__(self, data_dir='./', batch_size=32):
        """_summary_

        Args:
            data_dir (str, optional): _description_. Defaults to './'.
            batch_size (int, optional): _description_. Defaults to 32.
        """        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])

    def prepare_data(self):
        """_summary_
        """        
        # 데이터셋 다운로드
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """_summary_

        Args:
            stage (_type_, optional): _description_. Defaults to None.
        """        
        # 데이터셋 로드 및 분할
        if stage in (None, 'fit'):
            cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
        if stage in (None, 'test'):
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        # 학습 데이터 로더
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        # 검증 데이터 로더
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        # 테스트 데이터 로더
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

# 데이터 모듈과 모델 초기화
data_module = CIFAR10DataModule(data_dir='./', batch_size=32)
model = ResNet(num_classes=10)

# 트레이너 초기화 및 학습, 테스트 실행
# progress_bar에서 출력값이 많아져서 커널이 죽는 거 같아서 학습과정을 제거하고 실행결과만 나오게 했습니다.
trainer = pl.Trainer(max_epochs=30, accelerator='gpu', devices=1 if torch.cuda.is_available() else None, enable_progress_bar=False)
trainer.fit(model, data_module)
trainer.test(model, data_module)
