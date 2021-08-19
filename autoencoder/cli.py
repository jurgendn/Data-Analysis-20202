from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import CIFAR10Data
from model import AutoEncoderModel

import numpy as np
from utils import pickling_dataset

class AutoEncoderTrainingCLI(LightningCLI):
    """
    CLI for training AutoEncoder model
    """
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--pickle_embedd', action='store_true',
                            help="Pickle embedding vector and test dataset for clustering.")

    def add_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        save_checkpoint = ModelCheckpoint(dirpath=None,
                                          filename='{epoch}-{step}-{val_loss:.2f}',
                                          save_top_k=-1)
        self.config_init['trainer']['callbacks'] = [lr_monitor, save_checkpoint]

    def instantiate_trainer(self):
        self.add_callbacks()
        super().instantiate_trainer()

    def instantiate_model(self):
        temp_train_loader = self.datamodule.train_dataloader()
        train_loader_total_samples = len(temp_train_loader)
        import ipdb; ipdb.set_trace()
        self.config_init['model']['train_loader_total_samples'] = train_loader_total_samples
        super().instantiate_model()

    def after_fit(self):
        if self.config['pickle_embedd']:
            self.datamodule.setup('test')
            pickling_dataset(self.datamodule.dataset_test)
            
            res = self.trainer.predict(model=self.model, dataloaders=self.datamodule.test_dataloader())
            concat_res = np.concatenate(res, axis=0)
            np.save('../clustering/data/codes.npy', concat_res)

def cli_main():
    AutoEncoderTrainingCLI(AutoEncoderModel, CIFAR10Data,
                            seed_everything_default=42, save_config_callback=None)

if __name__ == "__main__":
    cli_main()