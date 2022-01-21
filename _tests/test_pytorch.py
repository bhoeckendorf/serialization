import torch
import pickle
import serialization
from copy import deepcopy
import pytest


def test_pytorch():
    arr = torch.ones((128, 256, 256), dtype=torch.float32)
    arr[8, 8, 8] = 2
    arrpkl = pickle.dumps(arr)

    data = [
        1, "2", arr,
        [3, "4", deepcopy(arr), {"arr": deepcopy(arr)}],
        {4: "5", "6": 7, "arr": deepcopy(arr), "lst": [deepcopy(arr)]}
    ]
    datapkl = pickle.dumps(data)

    for key in (None, "key".encode()):
        ser = serialization.serialize(arr, key)
        assert len(ser) < 0.5 * len(arrpkl)
        deser = serialization.deserialize(ser, key)
        assert torch.all(arr == deser)

        ser = serialization.serialize(data, key)
        assert len(ser) < len(datapkl)
        assert len(ser) < len(arrpkl)
        deser = serialization.deserialize(ser, key)
        assert torch.all(deser[2] == arr)
        assert torch.all(deser[3][2] == arr)
        assert torch.all(deser[3][3]["arr"] == arr)
        assert torch.all(deser[4]["arr"] == arr)
        assert torch.all(deser[4]["lst"][0] == arr)
    
    with pytest.raises(RuntimeError):
        serialization.deserialize(serialization.serialize(arr, key), "nokey".encode())
