import os
import json
from copy import deepcopy
try:
    import pyedflib  # type: ignore
except Exception:  # optional dependency (only needed for EDF)
    pyedflib = None

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset


class CauEegDataset(Dataset):


    def __init__(self, root_dir: str, data_list: list, load_event: bool, file_format: str = "edf", transform=None):
        if file_format not in ["edf", "feather", "memmap", "np"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )

        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.file_format = file_format
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

            # 获取原始数据（保持原始格式）
        sample = deepcopy(self.data_list[idx])
        sample["signal"] = self._read_signal(sample)

        if self.load_event:
            sample["event"] = self._read_event(sample)

        # 先应用transform（transform仍操作sample['signal']）
        if self.transform:
            sample = self.transform(sample)

        return sample  # 保持原始格式返回




    def _read_signal(self, anno):
        if self.file_format == "edf":
            return self._read_edf(anno)
        elif self.file_format == "feather":
            return self._read_feather(anno)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        if pyedflib is None:
            raise ModuleNotFoundError(
                "未安装 pyedflib，无法读取 EDF。请安装 pyedflib 或将 file_format 设为 feather/memmap。"
            )
        edf_file = os.path.join(self.root_dir, f"signal/edf/{anno['serial']}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    def _read_feather(self, anno):
        fname = os.path.join(self.root_dir, f"signal/feather/{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_memmap(self, anno):
        fname = os.path.join(self.root_dir, f"signal/memmap/{anno['serial']}.dat")
        signal = np.memmap(fname, dtype="int32", mode="r").reshape(21, -1)
        return signal

    def _read_np(self, anno):
        fname = os.path.join(self.root_dir, f"signal/{anno['serial']}.npy")
        return np.load(fname)

    def _read_event(self, m):
        fname = os.path.join(self.root_dir, "event", m["serial"] + ".json")
        with open(fname, "r") as json_file:
            event = json.load(json_file)
        return event
