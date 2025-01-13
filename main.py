    
    
    
return init_net(netD, 'normal', 0.02, device)

netD = networks.define_D(opt.input_nc, opt.ndf, self.opt.device) # Discriminator for netG
netD2 = networks.define_D(opt.input_nc - 2, opt.ndf, self.opt.device) # Discriminator for netLBP
