# torch-inductor-duplicate-index-writeorder-guard

This repository has been consolidated into **torch-correctness-guards**.

Use the umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git

torch-guard run duplicate-index-writeorder
```

Python API:

```python
from torch_correctness_guards import safe_dup_index_assign

safe_dup_index_assign(v, idx, delta)
```

The migrated guard still diagnoses pytorch/pytorch#197582: an Inductor divergence for `v[idx] = v[idx] + delta` when `idx` is a computed duplicate index such as `torch.arange(n) % 2`.

This source repository is archived and kept only for historical reference. New fixes and tests belong in the umbrella package: https://github.com/zhuhroscar-tech/torch-correctness-guards
