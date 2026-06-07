#!/usr/bin/env python3
"""Regression + sanity tests for the Automagic2 optimizer and its guard.

Automagic2 fuses the optimizer step into ``register_post_accumulate_grad_hook``:
each parameter is updated and its ``.grad`` freed the instant autograd finishes
accumulating into it. ``.step()`` is therefore a no-op. This is fine for the
default (gradient_accumulation == 1) path, but it silently breaks the trainer's
``zero_grad -> backward (xN) -> clip_grad_norm_ -> step()`` contract when
gradient_accumulation > 1, because:

  * the weights update on EVERY microbatch backward instead of once per
    optimizer step (so GA does NOT average N microbatches into one step), and
  * ``clip_grad_norm_`` runs after grads are already None, so clipping is a
    silent no-op.

These tests pin that behaviour and the additive guard added in
``toolkit.optimizer.get_optimizer`` / the trainer.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest testing/test_automagic2.py
All tests are CPU-only and allocate no CUDA.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from toolkit.optimizers.automagic2 import Automagic2
from toolkit.optimizer import get_optimizer


def _cpu():
    # Belt-and-braces: never touch CUDA from this module.
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. The regression: step is fused into backward; .step() / clip are no-ops.
# ---------------------------------------------------------------------------

def test_param_updates_on_every_backward_not_step():
    """Each loss.backward() mutates p.data immediately and frees p.grad.

    Proves the step is fused into the backward hook (per-microbatch update),
    not accumulate-then-step. This is the mechanism that makes GA>1 wrong.
    """
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False).to(_cpu())
    # lr high enough that a single update is clearly visible in fp32.
    opt = Automagic2(lin.parameters(), lr=1e-3, max_lr=1e-2)

    x = torch.randn(4, 8)
    n_backwards = 4
    snapshots = [lin.weight.detach().clone()]
    for _ in range(n_backwards):
        # NOTE: deliberately NO opt.step() between backwards — mimics the
        # microbatch loop inside one trainer optimizer step under GA>1.
        loss = (lin(x) ** 2).mean()
        loss.backward()
        # Hook ran during backward: grad must already be freed.
        assert lin.weight.grad is None, "hook should set p.grad=None after backward"
        snapshots.append(lin.weight.detach().clone())

    # p.data changed on EVERY backward, not just once.
    for i in range(1, len(snapshots)):
        delta = (snapshots[i] - snapshots[i - 1]).abs().max().item()
        assert delta > 0.0, f"weight did not change on backward #{i} (delta={delta})"


def test_clip_grad_norm_is_noop_after_fused_step():
    """Once the hook frees grads, clip_grad_norm_ sees nothing -> returns 0.0.

    This is the exact ordering the trainer uses (backward then clip), so under
    automagic2 max_grad_norm is silently ineffective.
    """
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False).to(_cpu())
    params = list(lin.parameters())
    opt = Automagic2(params, lr=1e-3, max_lr=1e-2)

    loss = (lin(torch.randn(4, 8)) ** 2).mean()
    loss.backward()
    # All grads are None post-backward (freed by the fused hook).
    assert all(p.grad is None for p in params)
    total_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert float(total_norm) == 0.0, (
        f"clip_grad_norm_ should be a no-op (0.0) after grads freed, got {float(total_norm)}"
    )


def test_step_is_noop_on_params():
    """opt.step() must return without changing params (no grads, fused design)."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False).to(_cpu())
    opt = Automagic2(lin.parameters(), lr=1e-3, max_lr=1e-2)

    loss = (lin(torch.randn(4, 8)) ** 2).mean()
    loss.backward()  # fused update happens here
    before = lin.weight.detach().clone()
    ret = opt.step()
    after = lin.weight.detach().clone()
    assert ret is None
    assert torch.equal(before, after), "step() must not change params (it is a no-op)"


# ---------------------------------------------------------------------------
# 2. Positive sanity: the optimizer actually learns; lr API works.
# ---------------------------------------------------------------------------

def test_optimizer_reduces_quadratic_loss():
    """A handful of fused updates drive a trivial quadratic toward its minimum."""
    torch.manual_seed(0)
    # Minimise ||w - target||^2 over a free parameter w.
    w = nn.Parameter(torch.randn(16))
    target = torch.full((16,), 0.5)
    opt = Automagic2([w], lr=1e-3, max_lr=1e-2, lr_bump=1e-4)

    losses = []
    for _ in range(50):
        loss = ((w - target) ** 2).mean()
        losses.append(loss.item())
        loss.backward()  # fused step
        assert w.grad is None
    final = ((w - target) ** 2).mean().item()
    losses.append(final)
    assert final < losses[0], f"loss did not decrease: {losses[0]:.5f} -> {final:.5f}"


def test_get_avg_learning_rate_works():
    """get_avg_learning_rate returns a positive float within [min_lr, max_lr]."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False).to(_cpu())
    opt = Automagic2(lin.parameters(), lr=1e-4, min_lr=1e-7, max_lr=1e-3)
    # Before any step it falls back to the group/default lr.
    pre = opt.get_avg_learning_rate()
    assert isinstance(pre, float) and pre > 0.0
    for _ in range(5):
        loss = (lin(torch.randn(4, 8)) ** 2).mean()
        loss.backward()
    avg = opt.get_avg_learning_rate()
    assert isinstance(avg, float)
    assert 1e-7 <= avg <= 1e-3, f"avg lr {avg} outside [min_lr, max_lr]"


# ---------------------------------------------------------------------------
# 3. The guard: get_optimizer must reject automagic2 + gradient_accumulation>1.
# ---------------------------------------------------------------------------

def _linear_params():
    return list(nn.Linear(4, 4).parameters())


def test_guard_raises_on_automagic2_with_ga_gt_1():
    with pytest.raises(ValueError) as ei:
        get_optimizer(
            _linear_params(),
            optimizer_type="automagic2",
            learning_rate=1e-4,
            gradient_accumulation=2,
        )
    msg = str(ei.value).lower()
    assert "automagic2" in msg and "gradient_accumulation" in msg, (
        f"guard message should name automagic2 + gradient_accumulation, got: {ei.value!r}"
    )


def test_guard_allows_automagic2_with_ga_1():
    # GA == 1 is the supported/default path and must build normally.
    opt = get_optimizer(
        _linear_params(),
        optimizer_type="automagic2",
        learning_rate=1e-4,
        gradient_accumulation=1,
    )
    assert isinstance(opt, Automagic2)


def test_guard_allows_other_optimizer_with_ga_gt_1():
    # The guard must be scoped to automagic2 only — adam+GA=2 is fine.
    opt = get_optimizer(
        _linear_params(),
        optimizer_type="adamw",
        learning_rate=1e-4,
        gradient_accumulation=4,
    )
    assert isinstance(opt, torch.optim.AdamW)


def test_default_call_signature_unchanged_for_ga_1_default():
    """Omitting gradient_accumulation entirely (old call sites) still works."""
    opt = get_optimizer(_linear_params(), optimizer_type="automagic2", learning_rate=1e-4)
    assert isinstance(opt, Automagic2)


def test_max_grad_norm_warning_is_one_time(capsys):
    """automagic2 + max_grad_norm>0 warns once that clipping is ineffective."""
    get_optimizer(
        _linear_params(),
        optimizer_type="automagic2",
        learning_rate=1e-4,
        gradient_accumulation=1,
        max_grad_norm=1.0,
    )
    out1 = capsys.readouterr().out.lower()
    assert "max_grad_norm" in out1 and "automagic2" in out1, (
        f"expected a max_grad_norm/automagic2 warning, got: {out1!r}"
    )
    # Second construction in the same process must NOT re-warn (one-time).
    get_optimizer(
        _linear_params(),
        optimizer_type="automagic2",
        learning_rate=1e-4,
        gradient_accumulation=1,
        max_grad_norm=1.0,
    )
    out2 = capsys.readouterr().out.lower()
    assert "max_grad_norm" not in out2, f"warning should be one-time, re-fired: {out2!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
