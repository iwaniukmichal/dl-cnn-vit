from __future__ import annotations


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback path
    def tqdm(iterable=None, **kwargs):
        return iterable


def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs)
