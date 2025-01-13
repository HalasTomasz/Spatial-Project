    
    
import discriminator
import lbp_model
import losses
import torch
import torch.nn as nn
import generator


def train():

    
    # Create the discriminators
    discriminator_LBP = discriminator.define_discriminator(opt.input_nc - 2, opt.ndf, opt.device)
    discriminator_LBP = init_net(net_LBP, 'normal', 0.02, opt.device)

    discriminator_Gen = discriminator.define_discriminator(opt.input_nc, opt.ndf, opt.device)
    discriminator_Gen = init_net(net_Gen, 'normal', 0.02, opt.device)

    # Create the GAN loss
    gan_loss = losses.GanLoss(opt.gan_loss_type, target_real_label=1.0, target_fake_label=0.0)

    # Create the generators 
    generator_lpd = lbp_model.LBPGenerator(opt.LBP.ngf, opt.LBP.use_spectral_norm).to(opt.device)
    generator_gen = generator.ImageGenerator(opt.GEN.ngf, opt.GEN.use_spectral_norm, opt.device).to(opt.device)


    
def main():
    train()