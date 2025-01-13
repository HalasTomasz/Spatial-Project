import torch
import torch.nn as nn
import pytorch_lightning as pl


class LPD(pl.LightningModule):

    def __init__(self):
        super(LPD, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_config = None
        
        self.muli_level_loss = nn.MultiLevelLoss()
        self.reconstruction_loss = self.loss_fn(y_hat, y)
        self.adversal_loss = self.loss_fn(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def freeze_layers(self):
        """
        Set requires_grad to False for all parameters of the model.
        Customize as needed to freeze specific layers.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        print("All layers have been frozen.")
   
    def loss(self, y_hat, y):

        final_loss = (
            self.model_config['muli_level_loss_paramter'] * self.muli_level_loss +
            self.model_config['reconstruction_loss_parameter'] * self.reconstruction_loss +
            self.model_config['adversal_loss_paramter'] * self.adversal_loss
        )

        return final_loss