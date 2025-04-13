import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from dataloader import PairedDataset
from discriminator import PatchDiscriminator
from feature_extractor import VGGFeatureExtractor
from pretrained_resnet_encoder import CartoonGenerator

def train():
    # Directories (update as needed)
    face_dir = '/kaggle/input/comic-faces-paired-synthetic/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/face'
    comic_dir = '/kaggle/input/comic-faces-paired-synthetic/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/comics'

    # Image transformations: resize, center crop, and normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Resize(286),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = PairedDataset(face_dir, comic_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # Initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = CartoonGenerator(resnet_model='resnet18').to(device)
    netD = PatchDiscriminator().to(device)
    vgg_extractor = VGGFeatureExtractor().to(device)

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss weights
    content_weight = 10.0
    num_init_epochs = 10  # initialization phase epochs
    num_epochs = 50      # full training epochs

    # Loss function
    L1_loss = nn.L1Loss()

    # ----------
    # Initialization Phase: Train netG only with content loss
    # ----------
    print("Starting Initialization Phase...")
    netG.train()
    for epoch in range(num_init_epochs):
        for i, (face_img, comic_img) in enumerate(dataloader):
            face_img = face_img.to(device)
            # Forward pass through generator
            # Compute content loss using VGG features
            face_features = vgg_extractor(face_img)
            fake_features = vgg_extractor(fake_cartoon)
            content_loss = L1_loss(fake_features, face_features)
            loss = content_loss

            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[Init Epoch {epoch+1}/{num_init_epochs}] Batch {i}, Content Loss: {content_loss.item():.4f}")

    # ----------
    # Full Training Phase: Joint adversarial and content losses
    # ----------
    print("Starting Full Training Phase...")
    for epoch in range(num_epochs):
        for i, (face_img, comic_img) in enumerate(dataloader):
            face_img = face_img.to(device)
            comic_img = comic_img.to(device)
            # Generate fake cartoon from face
            fake_cartoon = netG(face_img)

            # Generate edge-smoothed version of real comic images
            comic_imgs_smoothed = []
            for img in comic_img:
                pil_img = transforms.ToPILImage()(img.cpu())
                smoothed = edge_smoothing(pil_img)
                tensor_smoothed = transform(smoothed)  # Apply same transform as input
                comic_imgs_smoothed.append(tensor_smoothed)
            comic_smoothed = torch.stack(comic_imgs_smoothed).to(device)

            # ------------------
            # Update Discriminator
            # ------------------
            netD.zero_grad()
            # Real cartoon loss
            pred_real = netD(comic_img)
            loss_real = torch.mean(torch.log(pred_real + 1e-8))
            # Edge-smoothed cartoon loss
            pred_edge = netD(comic_smoothed)
            loss_edge = torch.mean(torch.log(1 - pred_edge + 1e-8))
            # Fake cartoon loss
            pred_fake = netD(fake_cartoon.detach())
            loss_fake = torch.mean(torch.log(1 - pred_fake + 1e-8))
            loss_D = -(loss_real + loss_edge + loss_fake)*100
            loss_D.backward()
            optimizerD.step()

            # ------------------
            # Update Generator
            # ------------------
            netG.zero_grad()
            # Content loss
            face_features = vgg_extractor(face_img)
            fake_features = vgg_extractor(fake_cartoon)
            content_loss = L1_loss(fake_features, face_features)
            # Adversarial loss: fool discriminator
            pred_fake = netD(fake_cartoon)
            adv_loss = torch.mean(torch.log(1 - pred_fake + 1e-8))
            loss_G = content_weight * content_loss + adv_loss * 1000
            loss_G.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {i} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | Content: {content_loss.item():.4f} | Adv: {adv_loss.item():.4f}")

        # Optionally, save model checkpoints here
        torch.save(netG.state_dict(), f'netG_epoch_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'netD_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()
