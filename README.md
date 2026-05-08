# Latent Adversarial Detection (LAD)

> Original paper: https://arxiv.org/abs/2604.28129

Unfamiliar with a term? Check out [VOCAB.md](VOCAB.md)

Reproduction of **arXiv:2604.28129v1** — *Latent Adversarial Detection: Adaptive Probing
of LLM Activations for Multi-Turn Attack Detection* (Kulkarni, 2026).

LAD monitors an LLM's residual stream in real time and flags multi-turn prompt-injection
attacks before the first overtly harmful turn. It works by tracking *adversarial
restlessness* — the elevated cumulative activation drift that phase-shifted attacks
produce — using five scalar trajectory features and raw activation vectors fed into an
XGBoost probe (primary, Sections 6–7). An optional contrastive MLP encoder stage
(Section 3.4 / Appendix C) can replace the raw activations with 128-dim embeddings;
pass `--variant contrastive` to the training and inference scripts to use it.

## Documentation

| Doc | Contents |
|---|---|
| [prerequisites](docs/prerequisites.md) | Hardware, software, and account requirements |
| [quick-start](docs/quick-start.md) | Smoke-test instructions (consumer GPU + A100) |
| [pipeline](docs/pipeline.md) | 7 phase pipeline |
| [pipeline-small](docs/pipeline-small.md) | 5 phase pipeline with small models and pre-generated data |
| [reference](docs/reference.md) | Script table + repository structure |

## Citation

```bibtex
@article{kulkarni2026lad,
  title   = {Latent Adversarial Detection: Adaptive Probing of {LLM} Activations
             for Multi-Turn Attack Detection},
  author  = {Kulkarni, Prashant},
  journal = {arXiv preprint arXiv:2604.28129},
  year    = {2026}
}
```
