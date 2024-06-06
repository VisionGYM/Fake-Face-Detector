#Generator
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, kernel_size = 4):
        super().__init__()
        self.kernel_size = kernel_size

        def block(in_feat, out_feat, stride_s = 2, padding_s = 1, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size = kernel_size , stride=stride_s, padding=padding_s, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator = nn.Sequential(
            *block(latent_dim, 512, stride_s = 1, padding_s = 0),
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride = 2, padding = 1, bias= False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

# 예시로 사용할 latent_dim 설정
latent_dim = 100

# Generator 인스턴스 생성
generator = Generator(latent_dim)

# 랜덤 노이즈 벡터 생성
x= torch.randn((1, latent_dim, 1, 1))  # 배치 크기 1, latent_dim 크기 100

# Forward pass 테스트
generated_image = generator(x)

print(generated_image)  # 출력: torch.Size([1, 3, 64, 64])

#가중치 초기화 함수
def weights_init(m):

   classname = m.__class__.__name__
   if classname.find('Conv') != -1:

       nn.init.normal_(m.weight.data, 0.0, 0.02)
   elif classname.find('BatchNorm') != -1:

       nn.init.normal_(m.weight.data, 1.0, 0.02)
       nn.init.constant_(m.bias.data, 0)