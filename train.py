import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from lbp_model import LBPGenerator
from generator import ImageGenerator
from utli import VGG16FeatureExtractor, init_net, gram_matrix
from discriminator import define_discriminator
from losses import GanLoss, Discounted_L1
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
class TrainModel(pl.LightningModule):

    def __init__(self, opt):
        super(TrainModel, self).__init__()

        self.save_hyperparameters()
        self.opt = opt

        # Initializing networks
        self.netLBP = LBPGenerator(opt.LBP.ngf, opt.LBP.use_spectral_norm).to(opt.device)
        self.netG = ImageGenerator(opt.GEN.ngf, opt.GEN.use_spectral_norm, opt.device).to(opt.device)
        self.vgg16_extractor = VGG16FeatureExtractor().to(self.opt.device)

        self.netD2 = define_discriminator(opt.input_nc - 2, opt.ndf, opt.device)
        self.netD2 = init_net(self.netD2, 'normal', 0.02, opt.device)

        self.netD = define_discriminator(opt.input_nc, opt.ndf, opt.device)
        self.netD = init_net(self.netD, 'normal', 0.02, opt.device)

        # Initializing loss functions
        self.criterionGAN = GanLoss(loss_type=opt.gan_type).to(self.opt.device)
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        self.criterionL1_mask = Discounted_L1(opt).to(self.opt.device)
        self.criterionL2_style_loss = nn.MSELoss()
        self.criterionL2_perceptual_loss = nn.MSELoss()

        self.automatic_optimization = False  # Turn off automatic optimization

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def freeze_layers(self, model, parameter):
        for param in model.parameters():
            param.requires_grad = parameter
        print("All layers have been frozen.")
   
    def forward(self, input):
        Img_mask = input['img']
        Img_org = Img_mask.clone()

        self.lbp_mask = input['lbp_mask']
        self.lbp_org = self.lbp_mask.clone()

        self.mask = input['mask']
        self.mask_bytes = input['mask'].bool()

        Img_mask[:, :3].masked_fill_(self.mask_bytes, 0.)
        # Img_mask.narrow(1, 0, 1).masked_fill_(self.mask_bytes, 0.)
        # Img_mask.narrow(1, 1, 1).masked_fill_(self.mask_bytes, 0.)
        # Img_mask.narrow(1, 2, 1).masked_fill_(self.mask_bytes, 0.)
        
        self.lbp_mask.masked_fill_(self.mask_bytes, 0.)

        self.I_i = Img_mask
        self.I_g = Img_org

        self.L_o = self.netLBP(self.lbp_mask, self.mask)
        _, self.I_FEA = self.netG(self.I_g, self.lbp_org, self.mask)
        self.I_o, self.i_fea  = self.netG(self.I_i, self.L_o, self.mask)

    def training_step(self, batch, batch_idx):

        # G D D2
        opt1, opt2, opt3 = self.optimizers()  # Access both optimizers
        
        self.forward(batch)

        loss_G = self.backward_G()
        loss_D = self.backward_D()
        loss_D2 = self.backward_D2()


        self.freeze_layers(self.netD2, True)
        opt3.zero_grad()
        loss_D2 = self.backward_D2()
        loss_D2.backward()
        opt3.step()
        self.freeze_layers(self.netD2, False)

        self.freeze_layers(self.netD, True)
        opt2.zero_grad()
        loss_D = self.backward_D()
        loss_D.backward()
        opt2.step()
        self.freeze_layers(self.netD, False)

        opt1.zero_grad()
        loss_G = self.backward_G()
        loss_G.backward()
        opt1.step()

        # return self.I_g, self.I_o, self.loss_G

    def validation_step(self, batch, batch_idx):
        self.forward(batch)

        val_loss_G = self.backward_G()
        val_loss_D = self.backward_D()
        val_loss_D2 = self.backward_D2()

        self.log("val_loss_G", val_loss_G, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss_D", val_loss_D, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss_D2", val_loss_D2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        batch_length = len(batch['img']) 
        self.log('batch_length', batch_length) 
        # images = batch['img']  # Original images
        # masks = batch['mask']  # Ground truth masks
        # generated = self.I_o  # The generated images from the model

        # normalized_images = (images / 2) + 0.5  # Undo normalization
        # normalized_generated = (generated / 2) + 0.5
        
        # # Move tensors to CPU
        # images = normalized_images.cpu()
        # masks = masks.cpu()
        # generated = normalized_generated.cpu()

        # # Convert mask to 3 channels for visualization
        # masks_3ch = masks.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]

        # # Normalize mask to range [0, 1]
        # masks_normalized = masks_3ch / masks_3ch.max()

        # # Overlay mask on the original image
        # overlayed_images = images * 0.7 + masks_normalized * 0.3

        # # Concatenate for visualization: Original | Overlayed | Generated
        # img_grid = torch.cat((images, overlayed_images, generated), dim=0)  # [3*B, 3, H, W]
        # grid = make_grid(img_grid, nrow=images.size(0))  # nrow = batch size

        # # Plot the grid
        # plt.figure(figsize=(12, 6))
        # plt.imshow(grid.permute(1, 2, 0).numpy())
        # plt.axis('off')
        # plt.title("Original | Original + Mask | Generated")
        # plt.savefig(f"/net/pr2/projects/plgrid/plggthyroid/przestrzenne/val/grid_image_{batch_idx}.png", bbox_inches='tight') 

    def test_step(self, batch, batch_idx):
        self.forward(batch)

        test_loss_G = self.backward_G()
        test_loss_D = self.backward_D()
        test_loss_D2 = self.backward_D2()

        self.log("test_loss_G", test_loss_G, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss_D", test_loss_D, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss_D2", test_loss_D2, on_step=False, on_epoch=True, prog_bar=True)

            
        # images = batch['img']  # Original images
        # masks = batch['mask']  # Ground truth masks
        # generated = self.I_o  # The generated images from the model

        # normalized_images = (images / 2) + 0.5  # Undo normalization
        # normalized_generated = (generated / 2) + 0.5
        
        # # Move tensors to CPU
        # images = normalized_images.cpu()
        # masks = masks.cpu()
        # generated = normalized_generated.cpu()

        # # Convert mask to 3 channels for visualization
        # masks_3ch = masks.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]

        # # Normalize mask to range [0, 1]
        # masks_normalized = masks_3ch / masks_3ch.max()

        # # Overlay mask on the original image
        # overlayed_images = images * 0.7 + masks_normalized * 0.3

        # # Concatenate for visualization: Original | Overlayed | Generated
        # img_grid = torch.cat((images, overlayed_images, generated), dim=0)  # [3*B, 3, H, W]
        # grid = make_grid(img_grid, nrow=images.size(0))  # nrow = batch size

        # # Plot the grid
        # plt.figure(figsize=(12, 6))
        # plt.imshow(grid.permute(1, 2, 0).numpy())
        # plt.axis('off')
        # plt.title("Original | Original + Mask | Generated")
        # plt.savefig(f"/net/pr2/projects/plgrid/plggthyroid/przestrzenne/test/grid_image_{batch_idx}.png", bbox_inches='tight') 


    def backward_G(self):
        I_o = self.I_o
        I_g = self.I_g
        L_o = self.L_o
        L_g = self.lbp_org

        pred_I_o = self.netD(I_o)
        pred_L_o = self.netD2(L_o)
        loss_G_GAN = self.criterionGAN(pred_I_o, True) * self.opt.gan_weight
        loss_G_GAN += self.criterionGAN(pred_L_o, True) * self.opt.gan_weight

        loss_G_L2 = self.criterionL2(self.I_o, self.I_g) * 10
        loss_G_L2 += self.criterionL2(self.L_o, self.lbp_org) * self.opt.lambda_A

        vgg_ft_I_o = self.vgg16_extractor(I_o)
        vgg_ft_I_g = self.vgg16_extractor(I_g)
        loss_style = sum(
            self.criterionL2_style_loss(gram_matrix(vgg_ft_I_o[i]), gram_matrix(vgg_ft_I_g[i]))
            for i in range(3)
        ) * self.opt.style_weight

        loss_perceptual = sum(
            self.criterionL2_perceptual_loss(vgg_ft_I_o[i], vgg_ft_I_g[i])
            for i in range(3)
        ) * self.opt.content_weight

        # loss_multi = sum(
        #     self.criterionL2(self.I_fea[i], self.I_FEA[i]) * 0.01
        #     for i in range(len(self.I_fea))
        # )

        loss_G = loss_G_L2 + loss_G_GAN + loss_style + loss_perceptual
        self.log("loss_G", loss_G, on_step=True, sync_dist=True)
        return loss_G

    def backward_D(self):
        I_o = self.I_o
        I_g = self.I_g

        pred_I_o = self.netD(I_o.detach())
        pred_I_g = self.netD(I_g)

        loss_D = (self.criterionGAN(pred_I_o, False) + self.criterionGAN(pred_I_g, True)) * 0.5
        self.log("loss_D", loss_D, on_step=True, sync_dist=True)
        return loss_D

    def backward_D2(self):
        L_o = self.L_o
        L_g = self.lbp_org

        pred_L_o = self.netD2(L_o.detach())
        pred_L_g = self.netD2(L_g)

        loss_D2 = (self.criterionGAN(pred_L_o, False) + self.criterionGAN(pred_L_g, True)) * 0.5
        self.log("loss_D2", loss_D2, on_step=True, sync_dist=True)
        return loss_D2

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))

        return [optimizer_G, optimizer_D, optimizer_D2]
