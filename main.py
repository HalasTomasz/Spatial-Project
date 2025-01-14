    
    
import discriminator
import lbp_model
import losses
import torch
import torch.nn as nn
import generator
import yaml 
import os
from mask_generator import MaskGenerator
import matplotlib.pyplot as plt
from dataset import MyDataset
from torch.utils.data import DataLoader, random_split

class DotDict:
    """A dictionary that supports dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


def train(dataloader_train, dataloader_val, dataloader_test):

    
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

    ## Here TO DO 
    # create the optimizers for the generators and discriminators
    # lighting training
    # loss calculation
    # optimizer calculation
    # save the model chaec
    # make val 
    # make test
    # uplaod to compute


    
def prepare_data():

    print("Preparing data...")
    with open('./config/lpd_model.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
    opt = DotDict(yaml_data)

    if os.path.exists(opt.mask_root.global_mask):
        mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
        for root, _, files in os.walk(opt.data_root):
            for file in files:
                mask = mask_generator.wrapper_gmask()  # Generate the mask
                mask_path = os.path.join(opt.mask_root.global_mask, file)
                mask_generator.save_image(mask, mask_path)  
    
    if os.path.exists(opt.mask_root.random_sqaure_mask):
        mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
        for root, _, files in os.walk(opt.data_root):
            for file in files:
                mask, _, _ = mask_generator.create_rand_mask()  # Generate the mask
                mask_path = os.path.join(opt.mask_root.random_sqaure_mask, file)  
                mask_generator.save_image(mask, mask_path)  

    if os.path.exists(opt.mask_root.random_walk_mask):
        mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
        for _, _, files in os.walk(opt.data_root):
            for file in files:
                mask = mask_generator.create_random_walk_mask()
                mask_path = os.path.join(opt.mask_root.random_walk_mask, file)  
                mask_generator.save_image(mask, mask_path)  

    if os.path.exists(opt.data_root):

        dataset = MyDataset(opt)

        total_train_images = len(dataset)
        print(total_train_images)

        train_size = int(0.7 * total_train_images)  # 70% for training
        val_size = int(0.15 * total_train_images)  # 15% for validation
        test_size = total_train_images - train_size - val_size  # Remaining 15% for testing

        train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

        dataloader_train = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        dataloader_val = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        dataloader_test = DataLoader(test_subset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

        return dataloader_train, dataloader_val, dataloader_test
    else:
        raise FileNotFoundError("Data or mask directory not found.")

    
def main():
    dataloader_train, dataloader_val, dataloader_test = prepare_data()
    train(dataloader_train, dataloader_val, dataloader_test)

if __name__ == "__main__":
    main()