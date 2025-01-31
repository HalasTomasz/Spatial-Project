    
    
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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from train import TrainModel
import logging

class DotDict:
    """A dictionary that supports dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trainer_logger')


def train(dataloader_train, dataloader_val, dataloader_test, opt):
    """
    Train the TrainModel with given dataloaders.

    Args:
        dataloader_train (DataLoader): Training dataloader.
        dataloader_val (DataLoader): Validation dataloader.
        dataloader_test (DataLoader): Test dataloader.
        opt (Namespace): Configuration options for the model.

    Returns:
        None
    """

    # Initialize the model
    model = TrainModel(opt)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='loss_G',  # Monitors generator loss for saving the best model
        dirpath='checkpoints/',
        filename='best-checkpoint-{epoch:02d}-{loss_G:.4f}',
        save_top_k=1,
        mode='min'
    )

    logger.info("Starting the training")

    # Trainer
    trainer = Trainer(
        max_epochs=opt.epochs,
        devices=8,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        strategy="ddp",
    )
    logger.info(f"Train Start")

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # if dataloader_test:
    #     trainer.test(model, dataloaders=dataloader_test)
    
def prepare_data():

    print("Preparing data...")
    with open('/net/pr2/projects/plgrid/plggthyroid/przestrzenne/Spatial-Project/config/lpd_model.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
    opt = DotDict(yaml_data)

    ## DONE
    # if os.path.exists(opt.mask_root.global_mask):
    #     mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
    #     for root, _, files in os.walk(opt.data_root):
    #         for file in files:
    #             mask = mask_generator.wrapper_gmask()  # Generate the mask
    #             mask_path = os.path.join(opt.mask_root.global_mask, file)
    #             mask_generator.save_image(mask, mask_path)  
    
    # if os.path.exists(opt.mask_root.random_sqaure_mask):
    #     mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
    #     for root, _, files in os.walk(opt.data_root):
    #         for file in files:
    #             mask, _, _ = mask_generator.create_rand_mask()  # Generate the mask
    #             mask_path = os.path.join(opt.mask_root.random_sqaure_mask, file)  
    #             mask_generator.save_image(mask, mask_path)  

    # if os.path.exists(opt.mask_root.random_walk_mask):
    #     mask_generator = MaskGenerator(fine_size=256, overlap=10)
        
    #     for _, _, files in os.walk(opt.data_root):
    #         for file in files:
    #             mask = mask_generator.create_random_walk_mask()
    #             mask_path = os.path.join(opt.mask_root.random_walk_mask, file)  
    #             mask_generator.save_image(mask, mask_path)  

    if os.path.exists(opt.data_root):

        dataset = MyDataset(opt)

        total_train_images = len(dataset)
        logger.info(f"Total size: {total_train_images}")

        train_size = int(0.3 * total_train_images)  # 70% for training
        val_size = int(0.15 * total_train_images)  # 15% for validation
        test_size = total_train_images - train_size - val_size  # Remaining 15% for testing

        train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

        dataloader_train = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        dataloader_val = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        dataloader_test = DataLoader(test_subset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

        return dataloader_train, dataloader_val, dataloader_test, opt
    else:
        raise FileNotFoundError("Data or mask directory not found.")

    
def main():
    logger.info("Starting")
    dataloader_train, dataloader_val, dataloader_test, opt = prepare_data()
    logger.info("Data Loaded")

    train(dataloader_train, dataloader_val, dataloader_test, opt)

if __name__ == "__main__":
    main()