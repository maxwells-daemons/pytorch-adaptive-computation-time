tasks
=====

Includes data, code, and configuration to reproduce experiments from
`(Graves 2016) <https://arxiv.org/abs/1603.08983/>`_.
Each module includes a `torch.utils.data.IterableDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset/>`_ to generate data for that task.

Additionally, each module can be run as a script configured through command-line arguments.
In addition to those listed below, all `flags supported by pytorch-lightning's Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags/>`_ may be used.

For example, to run an experiment on one or more GPUs (recommended, as these take a long time),
use :code:`--gpus n` where n is the number of GPUs available, and similar with :code:`--tpu_cores`.

Parity
------

.. automodule:: tasks.parity

.. autoclass:: tasks.parity.ParityDataset

.. argparse::
    :module: tasks.parity
    :func: get_base_parser
    :prog: poetry run pytorch-adaptive-computation-time/tasks/parity.py

    By default, runs an easier version of the task. To reproduce the paper, use
    :code:`--bits 64 --hidden_size 128`.

Addition
--------

.. automodule:: tasks.addition

.. autoclass:: tasks.addition.AdditionDataset

.. argparse::
    :module: tasks.addition
    :func: get_base_parser
    :prog: poetry run pytorch_adaptive_computation_time/tasks/addition.py

    NOTE: uses a GRU instead of an LSTM, as originally used in the paper.
