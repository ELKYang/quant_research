# -*- coding: utf-8 -*-
"""
Created on 2022/10/27 11:15

@author: Yang Fan

对于Dataframe设计的dataloader, 与qlib解耦
处理的Dataframe格式如下
-------------------------------------------------------------------
TradingDay	SecuCode	feature1	feature2	... label
20170103	SH000300	-1.1218971	-1.0245758	...	-0.9714445
20170103	SH000903	-1.0386353	-1.0245758	...	-0.9088304
20170103	SH000905	-1.2590216	-1.0226022	...	-1.0765333
20170103	SH600000	-0.6082509	-0.5620975	...	-0.91596484
20170103	SH600004	-0.5814161	-1.0245758	...	-0.7749267
-------------------------------------------------------------------
"""
from copy import deepcopy
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class Dataset:
    def __init__(
        self,
        raw_data: pd.DataFrame,
        data_path: List[str],
        start: int,
        end: int,
        step_len: int = 20,
        fillna_type: str = "ffill+bfill",
    ):
        self.start = int(start)
        self.end = int(end)
        self.step_len = step_len
        assert (
            len(data_path) == 2
        ), "There should be 2 terms in data_path, data and label respectively."
        # read the data and set the TradingDay and instrument as SecuCode.
        self.data = raw_data
        self.data.query(
            f"{self.start - 1}<TradingDay<{self.end + 1}", inplace=True
        )
        # Truncate stock data that retains start and end times to save memory and speed up processing.
        self.data.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
        self.label = pd.read_hdf(data_path[1], key="data")
        self.label.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
        # To avoid the samples for labels that do not match the samples for features.
        self.data = self.data.join(self.label, how="left")
        self.data.sort_index(axis=0, inplace=True)
        kwargs = {"object": self.data}

        self.data_arr = np.array(**kwargs)
        self.data_arr = np.append(
            self.data_arr,
            np.full(
                (1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype
            ),
            axis=0,
        )
        # NOTE:
        # - Get index from numpy.array will much faster than DataFrame.values
        # - append last line with full NaN for better performance in `__getitem__`
        # - Keep the same dtype will result in a better performance
        self.nan_idx = -1  # The last line is all NaN
        self.fillna_type = fillna_type

        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)
        self.idx_map = self.idx_map2arr(self.idx_map)
        self.idx_arr = np.array(
            self.idx_df.values, dtype=np.float64
        )  # for better performance

        self.start_idx, self.end_idx = self.data_index.slice_locs(
            start=self.start, end=self.end
        )
        del self.data, self.label  # save memory

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe with <datetime, DataFrame>

        Returns
        -------
        Tuple[pd.DataFrame, dict]:
            1) the first element:  reshape the original index into a <datetime(row), instrument(column)> 2D dataframe
                SecuCode SH600000 SH600004 SH600006 SH600007 SH600008 SH600009  ...
                TradingDay
                20210111        0        1        2        3        4        5  ...
                20210112     4146     4147     4148     4149     4150     4151  ...
                20210113     8293     8294     8295     8296     8297     8298  ...
                20210114    12441    12442    12443    12444    12445    12446  ...
            2) the second element:  {<original index>: <row, col>}
        """
        idx_df = pd.Series(
            range(data.shape[0]), index=data.index, dtype=object
        )
        idx_df = idx_df.unstack().sort_index(axis=0).sort_index(axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @staticmethod
    def idx_map2arr(idx_map):
        # pytorch data sampler will have better memory control without large dict or list
        # - https://github.com/pytorch/pytorch/issues/13243
        # - https://github.com/airctic/icevision/issues/613
        # So we convert the dict into int array.
        # The arr_map is expected to behave the same as idx_map

        dtype = np.int32
        # set a index out of bound to indicate the none existing
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @property
    def empty(self):
        return len(self) == 0

    def get_index(self):
        """
        Get the pandas index of the data, it will be useful in following scenarios
        - Special sampler will be used (e.g. user want to sample day by day)
        """
        return self.data_index[self.start_idx : self.end_idx]

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        get series indices of self.data_arr from the row, col indices of self.idx_df

        Parameters
        ----------
        row : int
            the row in self.idx_df
        col : int
            the col in self.idx_df

        Returns
        -------
        np.array:
            The indices of data of the data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate(
                [np.full((self.step_len - len(indices),), np.nan), indices]
            )

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[Any, Any]:
        """
        get the col index and row index of a given sample index in self.idx_df

        Parameters
        ----------
        idx :
            the input of  `__getitem__`

        Returns
        -------
        Tuple[int]:
            the row and col index
        """
        # The the right row number `i` and col number `j` in idx_df
        real_idx = self.start_idx + idx
        if self.start_idx <= real_idx < self.end_idx:
            i, j = self.idx_map[real_idx]
        else:
            raise KeyError(
                f"{real_idx} is out of [{self.start_idx}, {self.end_idx})"
            )
        return i, j

    def __getitem__(self, idx):
        if isinstance(idx, list) and isinstance(idx[0], np.ndarray):
            result = []
            for i in range(len(idx)):
                indices_tmp = [
                    self._get_indices(*self._get_row_col(j)) for j in idx[i]
                ]
                indices_tmp = np.concatenate(indices_tmp)
                indices_tmp = np.nan_to_num(
                    indices_tmp.astype(np.float64), nan=self.nan_idx
                ).astype(int)
                data_tmp = self.data_arr[indices_tmp]
                data_tmp = data_tmp.reshape(
                    -1, self.step_len, *data_tmp.shape[1:]
                )
                result.append(data_tmp)
            return result
        else:
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
            indices = np.nan_to_num(
                indices.astype(np.float64), nan=self.nan_idx
            ).astype(int)
            data = self.data_arr[indices]
            data = data.reshape(-1, self.step_len, *data.shape[1:])
            return data

    def __len__(self):
        return self.end_idx - self.start_idx


class DailyBatchSampler(Sampler):
    def __init__(self, data_source, shuffle):
        super().__init__(data_source)
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = (
            pd.Series(index=self.data_source.get_index())
            .groupby(level=0)
            .size()
            .values
        )
        # calculate begin index of each batch
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = np.arange(len(self.daily_count))
            np.random.shuffle(indices)
            for idx in indices:
                yield np.arange(
                    self.daily_index[idx],
                    self.daily_index[idx] + self.daily_count[idx],
                )
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.daily_count)


def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]
