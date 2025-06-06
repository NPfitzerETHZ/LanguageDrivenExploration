# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import unravel_key
from torchrl.envs import Transform
import importlib

def _load_class(class_path: str):
    """
    Given a full class path string like 'torch.nn.modules.linear.Linear',
    dynamically load and return the class object.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load class '{class_path}': {e}")

def swap_last(source, dest):
    source = unravel_key(source)
    dest = unravel_key(dest)
    if isinstance(source, str):
        if isinstance(dest, str):
            return dest
        return dest[-1]
    if isinstance(dest, str):
        return source[:-1] + (dest,)
    return source[:-1] + (dest[-1],)


class DoneTransform(Transform):
    """Expands the 'done' entries (incl. terminated) to match the reward shape.

    Can be appended to a replay buffer or a collector.
    """

    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                tensordict.get(("next", done_key))
                .unsqueeze(-1)
                .expand(tensordict.get(("next", self.reward_key)).shape),
            )
        return tensordict

