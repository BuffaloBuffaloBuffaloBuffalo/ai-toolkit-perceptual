import torch

# Tracks whether the automagic2 max_grad_norm warning has already been emitted,
# so the heads-up prints once per process rather than once per optimizer rebuild.
_AUTOMAGIC2_CLIP_WARNED = False


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None,
        gradient_accumulation=1,
        max_grad_norm=None,
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()

    # Automagic2 fuses its update into register_post_accumulate_grad_hook: each
    # param is stepped and its .grad freed the instant autograd finishes the
    # backward, and .step() is a no-op. That breaks the trainer's
    # zero_grad -> backward(xN) -> clip_grad_norm_ -> step() contract whenever
    # gradient_accumulation > 1 (the weights would update on every microbatch
    # instead of once per step, and the clip would see freed grads). It also
    # makes max_grad_norm ineffective for the same reason. Guard the misuse
    # additively here; the gradient_accumulation == 1 default path is unchanged.
    if lower_type == 'automagic2':
        if gradient_accumulation is not None and gradient_accumulation > 1:
            raise ValueError(
                "optimizer 'automagic2' does not support gradient_accumulation > 1 "
                f"(got gradient_accumulation={gradient_accumulation}). automagic2 fuses "
                "the optimizer step into the backward pass (register_post_accumulate_grad_hook), "
                "so weights update on every microbatch instead of once per accumulated step and "
                "gradients are freed before they can be accumulated. Set gradient_accumulation=1, "
                "or choose a different optimizer."
            )
        global _AUTOMAGIC2_CLIP_WARNED
        if (max_grad_norm is not None and max_grad_norm > 0) and not _AUTOMAGIC2_CLIP_WARNED:
            print(
                "WARNING: max_grad_norm is ineffective with optimizer 'automagic2'. "
                "automagic2 frees gradients inside the backward pass (its step is fused "
                "into register_post_accumulate_grad_hook), so clip_grad_norm_ runs after the "
                "grads are already None and is a silent no-op."
            )
            _AUTOMAGIC2_CLIP_WARNED = True
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        import bitsandbytes

        if lower_type == "adam8bit":
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        if lower_type == "ademamix8bit":
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "adamw8bit":
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "lion8bit":
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic2':
        from toolkit.optimizers.automagic2 import Automagic2
        optimizer = Automagic2(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'rose':
        # Range-Of-Slice Equilibration optimizer (Kieren 2026, Apache 2.0).
        # Stateless: no per-param momentum/variance buffers; uses per-slice
        # |max| - min as the gradient denominator. See toolkit/optimizers/rose.py
        # for the full docstring + recommended hyperparams. LR must be tuned
        # independently — Adam defaults are NOT appropriate.
        from toolkit.optimizers.rose import Rose
        optimizer = Rose(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
