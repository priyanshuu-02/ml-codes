# IO-VNBD Dead Reckoning

This repository contains an IMU-based vehicle dead-reckoning research pipeline.

`Baseline V1` is preserved under the local model artefact store and is never overwritten by V7 training. V7 adds a causal initial-speed navigation state, balanced turn sampling, mild class weighting, and turn-aware longitudinal loss. It does **not** use the held-out test set for checkpoint selection.

## Train V7

```powershell
.\.venv\Scripts\python.exe train_v7.py --experiment v7_state_turn_aware
```

## Deployment gate

A checkpoint is a deployment candidate only after it improves validation and held-out test trajectory-outage metrics over V1, including 10/20/30/60-second final-position drift, while maintaining acceptable per-session and turning performance. Never use the test set to tune training settings.
