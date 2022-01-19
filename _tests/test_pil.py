import numpy as np
from PIL import Image 
import cloudpickle as pickle
import serialization
from copy import deepcopy
import pytest


def test_pil():
    arr = Image.radial_gradient("I")
    arrpkl = pickle.dumps(arr)

    data = [
        1, "2", arr,
        [3, "4", deepcopy(arr), {"arr": deepcopy(arr)}],
        {4: "5", "6": 7, "arr": deepcopy(arr), "lst": [deepcopy(arr)]},
        (3, "4", deepcopy(arr), {"arr": deepcopy(arr)})
    ]
    datapkl = pickle.dumps(data)

    for key in (None, "key".encode()):
        ser = serialization.serialize(arr, key)
        assert isinstance(arr, Image.Image)
        assert isinstance(ser, bytes)
        assert len(ser) < 0.5 * len(arrpkl)
        deser = serialization.deserialize(ser, key)
        assert isinstance(arr, Image.Image)
        assert isinstance(ser, bytes)
        assert isinstance(deser, Image.Image)
        assert np.all(np.asarray(arr) == np.asarray(deser))

        ser = serialization.serialize(data, key)
        assert isinstance(ser, bytes)
        assert isinstance(data[2], Image.Image)
        assert isinstance(data[3][2], Image.Image)
        assert isinstance(data[4]["arr"], Image.Image)
        assert len(ser) < len(datapkl)
        assert len(ser) < len(arrpkl)
        deser = serialization.deserialize(ser, key)
        assert isinstance(deser[2], Image.Image)
        assert isinstance(deser[3][2], Image.Image)
        assert isinstance(deser[4]["arr"], Image.Image)
        assert isinstance(deser[-1], tuple)
        assert np.all(np.asarray(deser[2]) == np.asarray(arr))
        assert np.all(np.asarray(deser[3][2]) == np.asarray(arr))
        assert np.all(np.asarray(deser[3][3]["arr"]) == np.asarray(arr))
        assert np.all(np.asarray(deser[4]["arr"]) == np.asarray(arr))
        assert np.all(np.asarray(deser[4]["lst"][0]) == np.asarray(arr))
    
    with pytest.raises(RuntimeError):
        serialization.deserialize(serialization.serialize(arr, key), "nokey".encode())
