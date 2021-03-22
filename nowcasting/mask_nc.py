# File to deal read and write the .mask extensions
import zlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from nowcasting.config import cfg
from netCDF4 import Dataset, num2date

_executor_pool = ThreadPoolExecutor(max_workers=16)


def read_mask_file(filepath, out):
    """Load mask file to numpy array

    Parameters
    ----------
    filepath : str
    out : np.ndarray

    Returns
    -------

    """
    with Dataset(filepath) as cur_nc:
        mask = cur_nc.variables["mask"][:]
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    out[:] = mask[cfg.CROP.X1 : cfg.CROP.X2, cfg.CROP.Y1 : cfg.CROP.Y2]
    return out


def save_mask_file(npy_mask, filepath):
    compressed_data = zlib.compress(npy_mask.tobytes(), 2)
    f = open(filepath, "wb")
    f.write(compressed_data)
    f.close()


def quick_read_masks(path_list, num):
    read_storage = np.ones(
        (num, cfg.SST.ITERATOR.HEIGHT, cfg.SST.ITERATOR.WIDTH), dtype=np.bool
    )
    # for i in range(num):
    #     read_storage[i] = read_mask_file(path_list[i])
    if not path_list:
        ret = read_storage.reshape(
            (num, 1, cfg.SST.ITERATOR.HEIGHT, cfg.SST.ITERATOR.WIDTH)
        )
    else:
        future_objs = []
        for i in range(num):
            obj = _executor_pool.submit(read_mask_file, path_list[i], read_storage[i])
            future_objs.append(obj)
        wait(future_objs)
        ret = read_storage.reshape(
            (num, 1, cfg.SST.ITERATOR.HEIGHT, cfg.SST.ITERATOR.WIDTH)
        )
    return ret
