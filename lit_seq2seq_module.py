import os

from argparse import ArgumentParser

from typing import Optional, Callable

from pytorch_lightning import LightningModule, Trainer
from torch import optim
import torch.nn as nn
from torch.optim import Optimizer

from chatbot_tutorial_code.chatbot_tutorial import build_model, calc_loss


class LitSeq2Seq(LightningModule):
    def __init__(
        self,
        attn_model="dot",
        hidden_size=500,
        encoder_n_layers=2,
        decoder_n_layers=2,
        dropout=0.1,
        teacher_forcing_ratio=1.0,
        clip=50.0,
        learning_rate=0.0001,
        decoder_learning_ratio=5.0,
        n_iteration=400,
        num_words = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.decoder, self.embedding, self.encoder = build_model(num_words,
            attn_model, decoder_n_layers, dropout, encoder_n_layers, hidden_size
        )

    def training_step(self, batch, batch_idx,optimizer_idx):
        assert optimizer_idx == 0 # for some stupid reason pytorch-lightning wants this optimizer_idx in manual optimization mode
        encoder_optimizer, decoder_optimizer = self.optimizers()

        input_variable, lengths, target_variable, mask, max_target_len = batch
        batch_size = input_variable.shape[1]

        loss, n_totals, print_losses = calc_loss(
            batch_size,
            self.encoder,
            self.decoder,
            input_variable,
            lengths,
            mask,
            max_target_len,
            target_variable,
            self.hparams.teacher_forcing_ratio,
        )
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hparams.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hparams.clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        log_dict = {'train_loss': loss}
        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}


    def validation_step(self, batch):
        input_variable, lengths, target_variable, mask, max_target_len = batch
        batch_size = input_variable.shape[1]

        loss, n_totals, print_losses = calc_loss(
            batch_size,
            self.encoder,
            self.decoder,
            input_variable,
            lengths,
            mask,
            max_target_len,
            target_variable,
            self.hparams.teacher_forcing_ratio,
        )
        return {'val_loss': loss}

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        dec_lr = self.hparams.decoder_learning_ratio

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=lr * dec_lr,
        )
        return encoder_optimizer, decoder_optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--data_dir', type=str, default=os.environ.get("DATA_DIR", "data"))
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--n_iteration', type=int, default=100)
        parser.add_argument('--num_words', type=int, default=None)
        return parser


