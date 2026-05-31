# Third-Party Code Notices

The Perceptual LoRA Toolkit is released under the MIT License (see [`LICENSE`](LICENSE)).
It bundles the third-party components listed below. Each one remains governed by its own
license, reproduced or linked here. Nothing in this file changes the MIT terms that cover
the rest of the project.

## Rose optimizer

- **Component:** Rose — Range-Of-Slice Equilibration optimizer
- **Author / copyright holder:** Matthew E. Kieren (Matthew Everet Kieren)
- **Upstream:** <https://github.com/MatthewK78/Rose> (`rose_opt.py`)
- **License:** Apache License 2.0 — full text at [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)
- **Vendored as:** `toolkit/optimizers/rose.py`

Rose is a stateless optimizer: it keeps no per-parameter state between steps (no momentum
or variance buffers, not even a step counter), rescaling gradients by a per-slice
`|max| - min` range instead of Adam's RMS denominator. It is exposed here via
`optimizer: rose` in training configs and as "Rose (stateless, experimental)" in the Web UI.

### Attribution

The author asks to be credited by name. From the upstream repository:

> If you use Rose in your research, project, or product, I would be grateful if you would
> mention it by name and credit its author, Matthew E. Kieren.

If you train with Rose and publish results, please credit **Matthew E. Kieren**. To cite
it formally:

```bibtex
@software{kieren2026rose,
  author    = {Kieren, Matthew E.},
  title     = {Rose: Range-Of-Slice Equilibration optimizer},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19589764},
  url       = {https://doi.org/10.5281/zenodo.19589764}
}
```

### Modifications

The source is vendored **verbatim** from upstream `main` (verified byte-for-byte identical
to `rose_opt.py`); the Apache 2.0 copyright and license header is preserved in the file.
The only change is the filename (`rose_opt.py` → `toolkit/optimizers/rose.py`). If the file
is modified in the future, note the change here and add an in-file "Modified from upstream"
notice, as required by the Apache License 2.0, Section 4(b).
