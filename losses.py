import torch.nn as nn
import torch
import numpy as np
class GanLoss(nn.Module):

    def __init__(self, loss_type='lsgan', target_real_label=1.0, target_fake_label=0.0):
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

        target_tensor = torch.tensor(1.0 if target_is_real else 0.0).to(prediction.device)
        target_tensor = target_tensor.expand_as(prediction) 
        
        if self.loss_type == 'wgan':
            return self.loss(prediction, target_tensor) 

        return self.loss(prediction, target_tensor)
    
class Discounted_L1(nn.Module):
    def __init__(self, opt):
        super(Discounted_L1, self).__init__()
        # Register discounting template as a buffer
        self.register_buffer('discounting_mask', torch.tensor(spatial_discounting_mask(opt.fineSize//2 - opt.overlap * 2, opt.fineSize//2 - opt.overlap * 2, 0.9, opt.discounting)))
        self.L1 = nn.L1Loss()

    def forward(self, input, target):
        self._assert_no_grad(target)
        input_tmp = input * self.discounting_mask
        target_tmp = target * self.discounting_mask
        return self.L1(input_tmp, target_tmp)


    def _assert_no_grad(self, variable):
        assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


def spatial_discounting_mask(mask_width, mask_height, discounting_gamma, discounting=1):
    """Generate spatial discounting mask constant.
    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Returns:
        tf.Tensor: spatial discounting mask
    """
    gamma = discounting_gamma
    shape = [1, 1, mask_width, mask_height]
    if discounting:
        # print('Use spatial discounting l1 loss.')
        mask_values = np.ones((mask_width, mask_height), dtype='float32')
        for i in range(mask_width):
            for j in range(mask_height):
                mask_values[i, j] = max(
                    gamma**min(i, mask_width-i),
                    gamma**min(j, mask_height-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 1)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape, dtype='float32')

    return mask_values