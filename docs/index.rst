Welcome to pytorch-adaptive-computation-time's documentation!
=============================================================

This library implements PyTorch modules for recurrent neural networks that can
learn to execute variable-time algorithms, as presented in
`Adaptive Computation Time for Recurrent Neural Networks (Graves 2016) <https://arxiv.org/abs/1603.08983/>`_.
These models can learn patterns requiring varying amounts of computation for
a fixed-size input, which is difficult or impossible for traditional neural networks.
The library aims to be clean, idiomatic, and extensible, offering a similar
interface to PyTorch's builtin recurrent modules.

The main features are:

 - A nearly drop-in replacement for
   `torch.nn.RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN/>`_- and
   `torch.nn.RNNCell <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell/>`_-style
   RNNs, but with the power of variable computation time.
 - A :class:`wrapper <models.AdaptiveCellWrapper>` which adds adaptive
   computation time to any RNNCell.
 - Data generators, configs, and training scripts to reproduce experiments from the paper.

Example
-------
Vanilla PyTorch GRU::

    rnn = torch.nn.GRU(64, 128, num_layers=2)
    output, hidden = rnn(inputs, initial_hidden)


GRU with adaptive computation time::

    rnn = models.AdaptiveGRU(64, 128, num_layers=2, time_penalty=1e-3)
    output, hidden, ponder_cost = rnn(inputs, initial_hidden)

Installation
------------

TODO


Contents
--------

All reusable modules are under :doc:`models`. Each file under :doc:`tasks` is a script
which reproduces a training task from
`Graves 2016 <https://arxiv.org/abs/1603.08983/>`_.

.. toctree::
   :maxdepth: 2

   models
   tasks

Code
----

TODO

License
-------

This code is licensed under the Apache License, Version 2.0.

BibTeX
------

You don't need to cite this code, but if it helps you in your research and you'd like to::

    @misc{swope2020ACT,
      title   = "pytorch-adaptive-computation-time",
      author  = "Swope, Aidan",
      journal = "GitHub",
      year    = "2020",
      url     = "TODO"
    }

If you use the experiment code, please also consider
`citing PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning#bibtex/>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
