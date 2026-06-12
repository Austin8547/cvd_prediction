import math
from torch.optim.lr_scheduler import LambdaLR
from configs import config

def get_warmup_cosine_scheduler(optimizer, warmup_epochs=config.WARMUP_EPOCHS, 
                                epochs=config.EPOCHS, eta_min=config.ETA_MIN, 
                                head_lr=config.HEAD_LR):
    """
    Constructs a Cosine Annealing Learning Rate scheduler with an initial linear warmup phase.
    """
    def warmup_cosine_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return eta_min / head_lr + 0.5 * (1.0 - eta_min / head_lr) * (
            1 + math.cos(math.pi * progress)
        )
    
    return LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
