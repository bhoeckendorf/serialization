from copy import deepcopy
import hashlib as hashlib
import pickle
from typing import Any, Optional, Callable

_PACK_FNS = None
_UNPACK_FNS = None

if _PACK_FNS is None:
    _PACK_FNS = []
    _UNPACK_FNS = []

    from ._numpy import get_pack_fn as _pack_np, get_unpack_fn as _unpack_np
    _PACK_FNS.append(_pack_np)
    _UNPACK_FNS.append(_unpack_np)

    try:
        from ._xarray import get_pack_fn as _pack_xr, get_unpack_fn as _unpack_xr
        _PACK_FNS.append(_pack_xr)
        _UNPACK_FNS.append(_unpack_xr)
    except ModuleNotFoundError:
        pass

    try:
        from ._pytorch import get_pack_fn as _pack_pt, get_unpack_fn as _unpack_pt
        _PACK_FNS.append(_pack_pt)
        _UNPACK_FNS.append(_unpack_pt)
    except ModuleNotFoundError:
        pass

    try:
        from ._pil import get_pack_fn as _pack_pl, get_unpack_fn as _unpack_pl
        _PACK_FNS.append(_pack_pl)
        _UNPACK_FNS.append(_unpack_pl)
    except ModuleNotFoundError:
        pass


def _get_pack_fn(x: Any) -> Optional[Callable]:
    for fn in _PACK_FNS:
        y = fn(x)
        if y is not None:
            return y
    return None


def _get_unpack_fn(x: Any) -> Optional[Callable]:
    for fn in _UNPACK_FNS:
        y = fn(x)
        if y is not None:
            return y
    return None


def pack(x, **kwargs):
    if isinstance(x, dict):
        return {k: pack(v, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [pack(i, **kwargs) for i in x]
    elif isinstance(x, tuple):
        return tuple(pack(i, **kwargs) for i in x)
    fn = _get_pack_fn(x)
    if fn is not None:
        return fn(x)
    return x


def unpack(x):
    if isinstance(x, dict):
        return {k: unpack(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [unpack(i) for i in x]
    elif isinstance(x, tuple):
        return tuple(unpack(i) for i in x)
    fn = _get_unpack_fn(x)
    if fn is not None:
        return fn(x)
    return x


def sign(x, key):
    return hashlib.blake2b(x, key=key).digest() + x


def verify(x, key):
    digest, x = x[:64], x[64:]
    if hashlib.blake2b(x, key=key).digest() != digest:
        raise RuntimeError("Cryptographic verification failed.")
    return x


def serialize(
        x,
        key,
        maxsize: Optional[int] = None,
        cname="blosclz",
        clevel=9,
        **kwargs
):
    x = pickle.dumps(pack(x, cname=cname, clevel=clevel, **kwargs))

    if maxsize is None:
        if key is None:
            return x
        return sign(x, key)

    parts = (x[i:i + maxsize] for i in range(0, len(x), maxsize))
    if key is None:
        return parts
    return [sign(i, key) for i in parts]


def deserialize(x, key):
    if isinstance(x, list):
        assert key is not None, "Must provide key"
        x = pickle.loads(b"".join(verify(i, key) for i in x))
    else:
        if key is not None:
            x = verify(x, key)
        x = pickle.loads(x)

    return unpack(x)
