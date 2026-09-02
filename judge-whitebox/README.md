# whitebox

Analysis tools for interpretability and ablation studies on LLM judges.

Run tests locally with `uv run pytest judge-whitebox/tests -q`. Full evaluations are user-launched on SWC.

Token-manifold methods use a separate Python 3.12 environment:

```bash
uv venv --python 3.12 .venv-manifold
uv pip install --python .venv-manifold/bin/python -r judge-whitebox/requirements-manifold.txt
MPLCONFIGDIR=/tmp/token-manifold-mpl LOKY_MAX_CPU_COUNT=8 \
  .venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_*.py -q
```

Run `token_manifold_record.py --estimate-only` and review its storage estimate against the ceph quota
before submitting `sbatch/token_manifold.sbatch`. The scripts never install packages on the gateway.
