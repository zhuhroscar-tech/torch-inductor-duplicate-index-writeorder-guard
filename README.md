# torch-inductor-duplicate-index-writeorder-guard

Guards a real `torch.compile(backend="inductor")` correctness bug
(**not** a general "duplicate index writes are undefined" case):
[pytorch/pytorch#197582](https://github.com/pytorch/pytorch/issues/197582).

## The bug

```python
import torch

def f(v):
    idx = torch.arange(4) % 2      # -> [0, 1, 0, 1], a COMPUTED duplicate index
    v[idx] = v[idx] + 1.0
    return v

v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f(v.clone()))                          # eager:    [2., 3., 3., 4.]

compiled = torch.compile(f, backend="inductor")
print(compiled(v.clone()))                   # inductor: [3., 4., 3., 4.]  <-- WRONG
```

Eager applies index 0's single logical read-modify-write exactly once
(last-write-wins for the repeated index) and index 1's write exactly
once. The `inductor` backend silently drops or reorders one of the two
logically-independent writes — **only when the duplicate index comes
from a computed expression** like `torch.arange(n) % 2`. The upstream
issue's own control case — the identical logical pattern with a
**literal** index tensor `torch.tensor([0, 1, 0, 1])` — does **not**
trigger the bug. This guard reproduces and demonstrates that isolation
directly (see `diagnose()`'s two cases), not just from prose.

This is deliberately distinct from PyTorch's documented "undefined
behavior" for `index_put_(accumulate=False)` with duplicate indices:
here eager and compiled produce **different, disagreeing** answers for
the exact same Python-level assignment syntax, which is a compiler
correctness bug regardless of which answer is "canonical."

## The guard

`safe_dup_index_assign(v, idx, delta)` wraps the read-modify-write
assignment in a function decorated with `torch._dynamo.disable`,
forcing a Dynamo graph-break so the assignment always executes eagerly
instead of being lowered through Inductor's fused kernel:

```python
from torch_inductor_duplicate_index_writeorder_guard import safe_dup_index_assign

def f(v):
    idx = torch.arange(4) % 2
    safe_dup_index_assign(v, idx, 1.0)
    return v

compiled = torch.compile(f, backend="inductor")
print(compiled(v.clone()))  # [2., 3., 3., 4.]  -- matches eager
```

This does not patch PyTorch; you replace the specific call site.

## CLI

```bash
pip install -e ".[torch]"
torch-inductor-duplicate-index-writeorder-guard        # human-readable
torch-inductor-duplicate-index-writeorder-guard --json # machine-readable
```

Exit code `0` if the guard fully restores eager semantics on this
host's installed torch build, `1` if not, `2` if torch isn't
installed.

## Verification

- Reproduced from scratch on this host (torch 2.14.0, CPU-only,
  `backend="inductor"`) before this repo was written — see
  `tests/test_core.py`.
- The literal-index control case is exercised alongside the triggering
  computed-index case in the same `diagnose()` call, so the isolation
  claim is testable, not asserted.
- The guard's correctness is verified by comparing against eager on
  the *same* triggering case where the native path is confirmed to
  diverge first — a guard that "matches" on a case that was never
  broken proves nothing.
- CI runs the full suite on `ubuntu-latest` and `macos-latest` (Python
  3.10 and 3.12) via GitHub Actions.

## Limitations

- Requires PyTorch >= 2.0 with the Inductor backend; CPU-only testing
  (no CUDA-specific code paths are exercised or claimed).
- This guard is a call-site workaround, not a PyTorch patch. If a
  future PyTorch release fixes #197582, `tests/test_core.py`'s
  bug-reproduction tests fail loudly (by design) with a comment
  pointing back to this README — that is the expected signal to relax
  or retire the guard, not a regression in this repo.
- Only the specific `v[idx] = v[idx] + delta` read-modify-write shape
  named in the upstream issue is covered; other advanced-indexing
  patterns are out of scope.
