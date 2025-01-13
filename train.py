

class LBP(pl.LightningModule):

    def __init__(self, model_lbp, model_gen, models_config, discrimantor_lbp, discrimantor_gen, gan_loss):
        super(LPD, self).__init__()

        self.model_lbp = model_lbp
        self.model_gen = model_gen

        self.models_config = models_config
        self.discrimantor_lbp = discrimantor_lbp
        self.discrimantor_gen = discrimantor_gen


        self.muli_level_loss = nn.MultiLevelLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.adversal_loss = gan_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        L_with_hole, mask = batch
        L_filled_gen = self.forward(L_with_hole)
        
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
   
    def loss(self, input_I, output_I_o):

        pred_I_o = self.discrimantor(output_I_o)

        vgg_ft_I_o = self.vgg16_extractor(I_o)

        final_loss = (
            self.model_config['muli_level_loss_paramter'] * self.muli_level_loss +
            self.model_config['reconstruction_loss_parameter'] * self.reconstruction_loss(input_I, output_I_o) +
            self.model_config['adversal_loss_paramter'] * self.adversal_loss(pred_I_o, True)
        )

        return final_loss