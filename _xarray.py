from copy import deepcopy
from typing import Any, Optional, Union, Callable

import xarray as xr

# from .serialization import pack as _pack, unpack as _unpack
from ._numpy import get_pack_fn as _get_pack_fn, get_unpack_fn as _get_unpack_fn


def get_pack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        return pack
    return None


def get_unpack_fn(x: Any) -> Optional[Callable]:
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        return unpack
    return None


def pack(x: Union[xr.DataArray, xr.Dataset], **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    x = deepcopy(x)
    if isinstance(x, xr.Dataset):
        for v in x.values():
            fn = _get_pack_fn(v._variable._data)
            if fn is not None:
                v._variable._data = fn(v._variable._data, **kwargs)
        #         _pack(v.attrs, **kwargs)
        # _pack(x.attrs, **kwargs)
        return x
    fn = _get_pack_fn(x._variable._data)
    if fn is not None:
        x._variable._data = fn(x._variable._data, **kwargs)
        # _pack(x.attrs, **kwargs)
    return x


def unpack(x: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    if isinstance(x, xr.Dataset):
        for v in x.values():
            fn = _get_unpack_fn(v._variable._data)
            if fn is not None:
                v._variable._data = fn(v._variable._data)
        #         _unpack(v.attrs)
        # _unpack(x.attrs)
        return x
    fn = _get_unpack_fn(x._variable._data)
    if fn is not None:
        x._variable._data = fn(x._variable._data)
    # _unpack(x.attrs)
    return x
