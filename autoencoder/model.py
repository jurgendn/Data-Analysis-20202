import numpy as np

from torch import optim
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from backbone import get_autoencoder


class AutoEncoderModel(pl.LightningModule):
    def __init__(self,
                model_name: str = 'dummy',
                optimizer_name: str = 'adam',
                lr_scheduler_method_name: str = 'none',
                lr: float = 0.001,
                momentum: float = 0.9,
                weight_decay: float = 5e-4,
                pretrained: bool = True,
                freeze_encoder: bool = False,
                checkpoint: str = None,
                train_loader_total_samples: int = 10000):
        """AutoEncoderModel
        Args:
        
        """
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint = checkpoint
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr_scheduler_method_name = lr_scheduler_method_name
        self.train_loader_total_samples = train_loader_total_samples
        self.inp_freeze_encoder = freeze_encoder
        
        self._build_model()
        self.criterion = nn.MSELoss()

    def _build_model(self):
        """
        Construct the model
        """
        self.model = get_autoencoder(model_name=self.model_name, pretrained=self.pretrained)
        if self.checkpoint is not None:
            try:
                self.load_from_checkpoint(self.checkpoint)
            except:
                print("Can't Load Checkpoint")
            else:
                print(f"Loaded checkpoint from {self.checkpoint}")

    def forward(self, x, only_encoder):
        """
        Model Forward Path
        """
        return self.model(x, only_encoder)
    
    def configure_optimizers(self):
        """
        Configuration of Optimizer and LR Scheduler
        """
        if self.inp_freeze_encoder:
            trainable_parameters = self.model.decoder.parameters()
        else:
            trainable_parameters = self.model.parameters()
        
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(trainable_parameters, lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer_name} is not defined')
        optimizer_list = [optimizer]

        if self.lr_scheduler_method_name == 'none':
            scheduler_list = []
        else:
            if self.lr_scheduler_method_name == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_loader_total_samples)
            elif self.lr_scheduler_method_name == 'cyclic':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=self.max_lr)
            else:
                raise ValueError(f"Learning rate scheduler {self.lr_scheduler_method_name} are not defined")
            scheduler_list = [scheduler]
        
        return optimizer_list, scheduler_list

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch of an training iteration.
        """
        image, _ = batch
        out = self.forward(image, only_encoder=False)
        loss = self.criterion(out, image)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch of a validation iteration.
        """
        image, _ = batch
        out = self.forward(image, only_encoder=False)
        loss = self.criterion(out, image)
        self.log('val_loss', loss, prog_bar=True)

    def on_predict_start(self):
        return super().on_predict_start()

    def predict_step(self, batch, batch_idx):
        """
        Predict step for each batch of a predict iteration.
        """
        image, label = batch
        codes = self.forward(image, only_encoder=True)

        # batch_size x 128 x 3 x 3
        np_codes = codes.cpu().detach().numpy().reshape(image.shape[0], -1)
        np_label = label.cpu().detach().numpy().reshape(image.shape[0], -1)
        res = np.hstack((np_codes, np_label))
        return res

    

    