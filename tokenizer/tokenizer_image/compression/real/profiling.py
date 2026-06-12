"""Runtime flag and profiling helpers for real AR generation."""

from __future__ import annotations

import os
import time

import torch


def _profile_add(profile, key, value):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(value)


def _profile_sync(device):
    if device is None:
        return
    if hasattr(device, "device"):
        device = device.device
    else:
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_tic(profile, device=None):
    if profile is None:
        return None
    _profile_sync(device)
    return time.perf_counter()


def _profile_toc(profile, key, start, device=None):
    if profile is None or start is None:
        return
    _profile_sync(device)
    _profile_add(profile, key, time.perf_counter() - start)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(name, default="0"):
    return str(os.environ.get(name, default)).strip().lower()
