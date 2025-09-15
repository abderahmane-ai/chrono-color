import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The original GAN paper uses a different loss function for the generator and discriminator.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the same size as the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.
        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def calculate_l1_loss(prediction, target):
    """Calculate L1 loss between prediction and target"""
    return F.l1_loss(prediction, target)


def calculate_psnr(prediction, target):
    """Calculate PSNR between prediction and target
    Assumes inputs are in [-1, 1] range from tanh activation
    """
    # Clamp values to ensure they're in [-1, 1] range
    prediction = torch.clamp(prediction, -1.0, 1.0)
    target = torch.clamp(target, -1.0, 1.0)
    
    # Convert from [-1, 1] to [0, 1] range for proper PSNR calculation
    prediction = (prediction + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    
    mse = F.mse_loss(prediction, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(prediction, target, size_average=True):
    """
    Calculate SSIM between prediction and target tensors.
    prediction, target: torch.Tensor of shape (N, C, H, W)
                        values should be in [0,1]
    size_average: if True, returns mean SSIM over the batch
    """
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    # ensure both are float tensors in [0,1]
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)

    ssim_value = ssim(prediction, target, data_range=1.0)

    if not size_average:
        return ssim_value.unsqueeze(0)
    return ssim_value