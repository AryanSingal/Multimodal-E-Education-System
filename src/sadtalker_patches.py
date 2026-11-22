"""Compatibility patches for running SadTalker with NumPy 2.x, PyTorch 2.x, and Librosa >= 0.10."""

import glob
import os
import re


SHIM_MARKER = "# === RUNTIME_COMPAT_SHIM ==="

SHIM_BLOCK = f"""{SHIM_MARKER}
import os
import warnings
import numpy as np
import torch

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = UserWarning

for _name, _builtin in [
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("complex", complex),
    ("object", object),
    ("str", str),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{{**kw, "weights_only": False}})

try:
    import librosa

    _orig_mel = librosa.filters.mel

    def _compat_mel(*args, **kwargs):
        if len(args) >= 2:
            kwargs.setdefault("sr", args[0])
            kwargs.setdefault("n_fft", args[1])
            args = args[2:]
        elif len(args) == 1:
            kwargs.setdefault("sr", args[0])
            args = ()
        return _orig_mel(*args, **kwargs)

    librosa.filters.mel = _compat_mel
except Exception:
    pass
"""


def patch_sadtalker_repo(sadtalker_root="/content/SadTalker"):
    """Apply NumPy, PyTorch, and Librosa compatibility fixes in-place."""
    if not os.path.exists(sadtalker_root):
        return

    # Patch 3D face preprocessing scalar casts
    face3d_prep = os.path.join(sadtalker_root, "src", "face3d", "util", "preprocess.py")
    if os.path.exists(face3d_prep):
        with open(face3d_prep, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("np.VisibleDeprecationWarning", "UserWarning")
        content = content.replace("float((t[0] - w0/2)*s)", "float(np.squeeze((t[0] - w0/2)*s))")
        content = content.replace("float((h0/2 - t[1])*s)", "float(np.squeeze((h0/2 - t[1])*s))")
        content = content.replace(
            "trans_params = np.array([w0, h0, s, t[0], t[1]])",
            "trans_params = np.array([float(np.squeeze(w0)), float(np.squeeze(h0)), "
            "float(np.squeeze(s)), float(np.squeeze(t[0])), float(np.squeeze(t[1]))])",
        )
        with open(face3d_prep, "w", encoding="utf-8") as f:
            f.write(content)

    # Patch utils/preprocess.py array unpacking
    utils_prep = os.path.join(sadtalker_root, "src", "utils", "preprocess.py")
    if os.path.exists(utils_prep):
        with open(utils_prep, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("np.VisibleDeprecationWarning", "UserWarning")
        content = content.replace(
            "[float(item) for item in np.hsplit(trans_params, 5)]",
            "[float(np.squeeze(item)) for item in np.hsplit(trans_params, 5)]",
        )
        with open(utils_prep, "w", encoding="utf-8") as f:
            f.write(content)

    # Patch librosa.filters.mel positional arguments in audio.py
    audio_py = os.path.join(sadtalker_root, "src", "utils", "audio.py")
    if os.path.exists(audio_py):
        with open(audio_py, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            "librosa.filters.mel(hp.sample_rate, hp.n_fft,",
            "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,",
        )
        with open(audio_py, "w", encoding="utf-8") as f:
            f.write(content)

    # Replace deprecated np.float / np.int aliases across the repository
    for py_file in glob.glob(os.path.join(sadtalker_root, "**", "*.py"), recursive=True):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            updated = txt.replace("np.VisibleDeprecationWarning", "UserWarning")
            updated = updated.replace(
                "[float(item) for item in np.hsplit(trans_params, 5)]",
                "[float(np.squeeze(item)) for item in np.hsplit(trans_params, 5)]",
            )
            updated = re.sub(r"\bnp\.float\b(?!32|64|16)", "float", updated)
            updated = re.sub(r"\bnp\.int\b(?!32|64|16|8|p|er)", "int", updated)
            if updated != txt:
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(updated)
        except OSError:
            pass

    # Prepend runtime shim to inference.py once
    inf_py = os.path.join(sadtalker_root, "inference.py")
    if os.path.exists(inf_py):
        with open(inf_py, "r", encoding="utf-8") as f:
            inf_txt = f.read()
        if SHIM_MARKER not in inf_txt:
            with open(inf_py, "w", encoding="utf-8") as f:
                f.write(SHIM_BLOCK + inf_txt)


if __name__ == "__main__":
    patch_sadtalker_repo()
    print("SadTalker compatibility patches applied.")
