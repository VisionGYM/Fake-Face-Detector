#Gan
class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def generator_step(self, batch):
        real_images, _ = batch
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, 100, 1, 1, device=self.device)

        fake_images = self.generator(noise)
        labels_real = torch.full((batch_size,), 1, device=self.device)

        output = self.discriminator(fake_images).view(-1)
        lossG = self.criterion(output, labels_real)

        return lossG

    def discriminator_step(self, batch):
        real_images, _ = batch
        batch_size = real_images.size(0)

        labels_real = torch.full((batch_size,), 1, device=self.device)
        labels_fake = torch.full((batch_size,), 0, device=self.device)

        output_real = self.discriminator(real_images).view(-1)
        lossD_real = self.criterion(output_real, labels_real)

        noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach()).view(-1)
        lossD_fake = self.criterion(output_fake, labels_fake)

        lossD = lossD_real + lossD_fake

        return lossD

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            lossD = self.discriminator_step(batch)
            self.log('loss_discriminator', lossD, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return lossD

        if optimizer_idx == 1:
            lossG = self.generator_step(batch)
            self.log('loss_generator', lossG, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return lossG

    def configure_optimizers(self):
        lr = 0.0002
        beta1 = 0.5
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        return [optimizerD, optimizerG], []

    def on_epoch_end(self):
        noise = torch.randn(64, 100, 1, 1, device=self.device)
        fake_images = self.generator(noise).detach().cpu()
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()
