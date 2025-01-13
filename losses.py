
class GanLoss(nn.Module):

    def __init__(self, loss_type='lsgan', target_real_label=1.0, target_fake_label=0.0)):
        super(GanLoss, self).__init__()

        self.loss_type = loss_type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if loss_type == 'wgan':
            self.loss = self.wgan_loss
        elif loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.loss = nn.BCELoss()
        #######################################################################
        ###  Relativistic GAN - https://github.com/AlexiaJM/RelativisticGAN ###
        #######################################################################
        # When Using `BCEWithLogitsLoss()`, remove the sigmoid layer in D.
        elif loss_type == 're_s_gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == 're_avg_gan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"GAN type {loss_type} not recognized.")

    def wan_loss(self, prediction, target_is_real):

        if target_is_real:
            return -prediction.mean()
        else:
            return prediction.mean()

    def forward(self, prediction, target_is_real):

        target_tensor = torch.tensor(1.0 if target_is_real else 0.0)
        target_tensor = target_tensor.expand_as(input)
        
        if self.loss_type == 'wgan':
            return self.loss(input, prediction)

        return self.loss(prediction, target_tensor)