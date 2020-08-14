#!/usr/bin/env python
"""
Implements the Parity task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

import argparse
import random
from typing import Tuple, Optional

import pytorch_lightning as pl
import torch

from pytorch_adaptive_computation_time import models


class ParityDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    An infinite IterableDataset for binary parity problems.

    Examples are pairs (vector, parity), where:
     - `vector` has a random number of places set to +/- 1 and the rest are
       zero.
     - `parity` is defined as 1 if there are an odd number of ones and
       0 otherwise.

    Parameters
    ----------
    bits
        The maximum length of each binary vector.
    """

    bits: int

    def __init__(self, bits: int):
        if bits <= 0:
            raise ValueError("bits must be at least one.")

        self.bits = bits

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = torch.zeros(self.bits, dtype=torch.float32)
        num_bits = random.randint(1, self.bits)
        bits = torch.randint(2, size=(num_bits,)) * 2 - 1
        vec[:num_bits] = bits
        parity = (bits == 1).sum() % 2
        return vec, parity.type(torch.float)


class ParityModel(pl.LightningModule):
    """
    An ACT RNN for parity tasks.
    """

    def __init__(
        self,
        bits: int,
        hidden_size: int,
        time_penalty: float,
        batch_size: int,
        learning_rate: float,
        time_limit: int,
        data_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = ParityDataset(bits)
        self.cell = models.AdaptiveRNNCell(
            input_size=bits,
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit,
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--max-steps", type=int, default=200000)
        parser.add_argument("--bits", type=int, default=16)
        parser.add_argument("--hidden_size", type=int, default=64)
        parser.add_argument("--time_penalty", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--time-limit", type=int, default=20)
        parser.add_argument("--data-workers", type=int, default=1)
        return parser

    def forward(
        self, binary_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, ponder_cost, steps = self.cell(binary_vector)
        logits = self.output_layer(hidden)
        return logits.squeeze(1), ponder_cost, steps

    def training_step(self, batch, _):
        vectors, targets = batch
        logits, ponder_cost, steps = self.forward(vectors)

        log_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss = log_loss + ponder_cost

        accuracy = (logits > 0).eq(targets).type(torch.float).mean().cpu()
        mean_steps = steps.type(torch.float).mean().cpu()

        return {
            "loss": loss,
            "log": {
                "loss/total": loss,
                "loss/classification": log_loss,
                "loss/ponder": ponder_cost,
                "accuracy": accuracy,
                "steps": mean_steps,
            },
        }

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
        )


def get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return ParityModel.add_argparse_args(parser)


def main():
    parser = get_base_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = ParityModel(
        bits=args.bits,
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_limit=args.time_limit,
        data_workers=args.data_workers,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, row_log_interval=100, max_steps=args.max_steps
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
