# RTGU — Recursive Transitional Gated Unit

Reference implementation for the Recursive Transitional Gated Unit (RTGU) introduced in:

> Nick van Kemenade. *Recursive Transitional Gated Unit (RTGU): A Novel Recurrent Architecture Demonstrating Empirical Superiority over LSTM*, 2025.

This repository contains:
- RTGU cell implementation.
- Comparative baselines (LSTM, GRU).
- Training and evaluation scripts.
- FLOPs and parameter count utilities (used in Section 3.3 of the paper).

---

## Installation

I recommend Python 3.10.

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# install dependencies
pip install -r requirements.txt
