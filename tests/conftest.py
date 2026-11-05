import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config  # noqa: E402


class CharTokenizer:
    """Offline stand-in tokenizer (byte-level) so tests never touch the network."""
    name_or_path = "char-test"
    eos_token_id = 256

    def __len__(self):
        return 257

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[min(ord(c), 255) for c in t] for t in texts]}


@pytest.fixture
def tok():
    return CharTokenizer()


@pytest.fixture
def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=257, n_positions=128, n_embd=32, n_layer=2, n_head=2))


@pytest.fixture
def tiny_cfg():
    cfg = load_config()
    cfg["data"].update(n_records=60, max_seq_len=64, val_fraction=0.2)
    cfg["lora"].update(dropout=0.0)
    cfg["train"].update(grad_accum=2, eval_every_steps=3, checkpoint_every_steps=3, max_eval_blocks=8,
                        eval_train_blocks=8, learning_rate=1e-3)
    cfg["wandb"]["mode"] = "disabled"
    cfg["hardware"]["num_threads"] = 1
    return cfg
