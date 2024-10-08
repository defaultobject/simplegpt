import jax
import jax.numpy as np
from jax import nn
import chex
import objax
import numpy as onp

import requests
import os
import matplotlib.pyplot as plt

from pathlib import Path

from gpt import SimpleTokenizer, Data, GPT, ADAM, progress_bar_callback
from gpt import sample

from jax import make_jaxpr

with jax.checking_leaks():

    # helper functions
    def download_tinyshakespeare(out_path: Path, name):
        out_path.mkdir(exist_ok=True)
        
        # see https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(out_path / name , 'w') as f:
            f.write(requests.get(data_url).text)
            
    def open_txt_file(file: Path):
        with open(file, 'r') as f:
            data = f.read()
        return data

    if True:
        download_tinyshakespeare(Path('data'), 'tiny_shakespeare.txt')
        data = open_txt_file(Path('data') / 'tiny_shakespeare.txt')
    else:
        data = open_txt_file(Path('data') / 'beatles.txt')

    block_size = 128
    batch_size = 64
    embedding_size = 192
    layers = 6
    num_heads = 6
    max_iters = 1000

    tokenizer = SimpleTokenizer()
    tokenizer.train(data)
    data_enc = tokenizer.encode(data)

    data_obj = Data(data_enc, block_size, batch_size)
    x, t = data_obj.batch()
    print(x.shape, t.shape)

    gpt = GPT(tokenizer.vocab_size, block_size, embedding_size, num_heads, layers, seed=0)

    trainer = ADAM(gpt)
    breakpoint()
    print(trainer.obj_fn(x, t))
    breakpoint()
    lc_arr = trainer.train(data_obj, 0.0005, max_iters, callback=progress_bar_callback(max_iters))

    objax.io.save_var_collection(f'model.npz', gpt.vars())

    print(sample('let it be', 1000, gpt, tokenizer, seed=0))

    plt.plot(lc_arr[10:]); plt.yscale('log'); plt.show()
