# torch-inductor-duplicate-index-writeorder-guard

本仓库已合并到 **torch-correctness-guards**。

请改用统一包：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git

torch-guard run duplicate-index-writeorder
```

Python API：

```python
from torch_correctness_guards import safe_dup_index_assign

safe_dup_index_assign(v, idx, delta)
```

迁移后的 guard 仍然诊断 pytorch/pytorch#197582：当 `idx` 是 `torch.arange(n) % 2` 这类计算得到的重复索引时，Inductor 会让 `v[idx] = v[idx] + delta` 与 eager 结果产生偏差。

本源仓库已归档，仅保留作历史参考。新的修复与测试应进入统一包：https://github.com/zhuhroscar-tech/torch-correctness-guards
