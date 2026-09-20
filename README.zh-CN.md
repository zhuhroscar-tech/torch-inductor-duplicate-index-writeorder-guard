[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-duplicate-index-writeorder-guard

针对一个真实的 `torch.compile(backend="inductor")` 正确性缺陷（并非泛泛的“重复索引写入未定义行为”）提供调用点级别的诊断与规避方案。上游追踪：[pytorch/pytorch#197582](https://github.com/pytorch/pytorch/issues/197582)。

## 缺陷

```python
import torch

def f(v):
    idx = torch.arange(4) % 2      # -> [0, 1, 0, 1]，由计算表达式产生的重复索引
    v[idx] = v[idx] + 1.0
    return v

v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f(v.clone()))                          # eager:    [2., 3., 3., 4.]

compiled = torch.compile(f, backend="inductor")
print(compiled(v.clone()))                   # inductor: [3., 4., 3., 4.]  <-- 错误
```

Eager 模式下，索引 0 的读-改-写只应用一次（重复索引下的“最后写入生效”语义），索引 1 的写入也只应用一次。`inductor` 后端会**静默丢弃或打乱**这两次逻辑上独立的写入之一——但**仅当重复索引来自计算表达式**（如 `torch.arange(n) % 2`）时才会触发。上游 issue 自带的对照用例——完全相同的逻辑模式，但索引换成**字面量张量** `torch.tensor([0, 1, 0, 1])`——则**不会**触发该缺陷。本仓库的 `diagnose()` 直接复现并展示这一隔离结果，而非仅凭文字断言。

这与 PyTorch 官方文档中"`index_put_(accumulate=False)` 配合重复索引属于未定义行为"的说法有本质区别：这里 eager 与编译模式对**完全相同的 Python 赋值语法**给出了**不一致**的结果，无论哪个答案才是"标准答案"，这都是一个真实的编译器正确性缺陷。

## 规避方案

`safe_dup_index_assign(v, idx, delta)` 用 `torch._dynamo.disable` 包装该读-改-写赋值，强制 Dynamo 在此处 graph-break，使赋值始终以 eager 方式执行，而不经过 Inductor 的融合内核：

```python
from torch_inductor_duplicate_index_writeorder_guard import safe_dup_index_assign

def f(v):
    idx = torch.arange(4) % 2
    safe_dup_index_assign(v, idx, 1.0)
    return v

compiled = torch.compile(f, backend="inductor")
print(compiled(v.clone()))  # [2., 3., 3., 4.]  -- 与 eager 一致
```

本方案不修改 PyTorch 本身，只替换特定调用点。

## 命令行工具

```bash
pip install -e ".[torch]"
torch-inductor-duplicate-index-writeorder-guard        # 人类可读
torch-inductor-duplicate-index-writeorder-guard --json # 机器可读
```

退出码：`0` 表示规避方案在当前主机安装的 torch 版本上完全恢复了 eager 语义，`1` 表示未恢复，`2` 表示未安装 torch。

## 验证

- 在编写本仓库代码之前，已在本机（torch 2.14.0，纯 CPU，`backend="inductor"`）从零复现——见 `tests/test_core.py`。
- 字面量索引对照用例与触发用例在同一次 `diagnose()` 调用中一并验证，因此"缺陷已隔离"的说法是可测试的，而非断言。
- 规避方案的正确性验证方式：先确认原生路径在同一触发用例上确实偏离 eager 结果，再验证规避方案与 eager 一致——在从未出问题的用例上"匹配"毫无意义。
- CI 在 GitHub Actions 的 `ubuntu-latest` 与 `macos-latest`（Python 3.10 与 3.12）上运行完整测试套件。

## 局限性

- 需要 PyTorch >= 2.0 且启用 Inductor 后端；仅测试了 CPU 路径（未覆盖或声称任何 CUDA 专属代码路径）。
- 本方案是调用点级别的规避，而非 PyTorch 补丁。若未来版本修复了 #197582，`tests/test_core.py` 中的缺陷复现测试会按设计明确失败，并在注释中指回本 README——这是应当放宽或退役该规避方案的信号，而非本仓库的回归。
- 仅覆盖上游 issue 中明确提到的 `v[idx] = v[idx] + delta` 读-改-写模式，其他高级索引模式不在本方案覆盖范围内。
