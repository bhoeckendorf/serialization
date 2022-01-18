from copy import deepcopy
import hashlib as hashlib
from typing import Any, Optional, Callable

import cloudpickle as pickle

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
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _pack(x, i, v, **kwargs)
    elif isinstance(x, dict):
        for i, v in x.items():
            _pack(x, i, v, **kwargs)
    fn = _get_pack_fn(x)
    if fn is not None:
        return fn(x)
    return x


def _pack(x, i, v, **kwargs):
    if isinstance(v, (dict, list)):
        pack(v, **kwargs)
    elif isinstance(v, tuple):
        x[i] = list(v)
        pack(x[i], **kwargs)
    fn = _get_pack_fn(v)
    if fn is not None:
        x[i] = fn(v)


def unpack(x):
    if isinstance(x, list):
        for i, v in enumerate(x):
            _unpack(x, i, v)
    elif isinstance(x, dict):
        for i, v in x.items():
            _unpack(x, i, v)
    fn = _get_unpack_fn(x)
    if fn is not None:
        return fn(x)
    return x


def _unpack(x, i, v):
    if isinstance(v, (dict, list)):
        unpack(v)
    fn = _get_unpack_fn(v)
    if fn is not None:
        x[i] = unpack(v)


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
    x = deepcopy(x)

    if isinstance(x, tuple):
        x = list(x)

    if isinstance(x, (dict, list)):
        pack(x, cname=cname, clevel=clevel, **kwargs)
    else:
        x = pack(x, cname=cname, clevel=clevel, **kwargs)
    x = pickle.dumps(x)

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

    if isinstance(x, (dict, list)):
        unpack(x)
        return x
    return unpack(x)
