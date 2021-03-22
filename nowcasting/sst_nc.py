# Python plugin that supports loading batch of images in parallel
import cv2
import numpy
import threading
from netCDF4 import Dataset, num2date
import os
import struct
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait
from nowcasting.config import cfg
import numpy as np

_imread_executor_pool = ThreadPoolExecutor(max_workers=16)


class UnknownImageFormat(Exception):
    pass


def quick_imsize(file_path):
    """Return (width, height) for a given nc file content - no external
    dependencies except the os and struct modules from core

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    size = os.path.getsize(file_path)
    with open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ("GIF87a", "GIF89a"):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif (
            (size >= 24)
            and data.startswith("\211PNG\r\n\032\n")
            and (data[12:16] == "IHDR")
        ):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith("\211PNG\r\n\032\n"):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith("\377\330"):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while b and ord(b) != 0xDA:
                    while ord(b) != 0xFF:
                        b = input.read(1)
                    while ord(b) == 0xFF:
                        b = input.read(1)
                    if ord(b) >= 0xC0 and ord(b) <= 0xC3:
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height


def read_nc_resize(path, read_storage, resize_storage, frame_size):
    with Dataset(path) as cur_nc:
        sst = cur_nc.variables["analysed_sst"]
        sst = sst.reshape(sst.shape[-2], sst.shape[-1])
    print("sst", sst)
    read_storage[:] = sst[cfg.CROP.X1 : cfg.CROP.X2, cfg.CROP.Y1 : cfg.CROP.Y2]
    resize_storage[:] = cv2.resize(
        read_storage, frame_size, interpolation=cv2.INTER_LINEAR
    )


def read_nc(path, read_storage):
    with Dataset(path) as cur_nc:
        sst = cur_nc.variables["analysed_sst"][:]
        sst = sst.reshape(sst.shape[-2], sst.shape[-1])
    read_storage[:] = sst[cfg.CROP.X1 : cfg.CROP.X2, cfg.CROP.Y1 : cfg.CROP.Y2]


def quick_read_frames(path_list, im_w=None, im_h=None, resize=False, frame_size=None):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    resize : bool, optional
    frame_size : None or tuple

    Returns
    -------

    """
    nc_num = len(path_list)
    # for i in range(nc_num):
    #     if not os.path.exists(path_list[i]):
    #         print("Here is the issue", path_list[i])
    #         raise IOError
    if im_w is None or im_h is None:
        im_w, im_h = quick_imsize(path_list[0])

    read_storage = numpy.empty((nc_num, im_h, im_w), dtype=numpy.uint8)

    if resize:
        resize_storage = numpy.empty(
            (nc_num, frame_size[0], frame_size[1]), dtype=numpy.uint8
        )
        if nc_num == 1:
            read_nc_resize(
                path=path_list[0],
                read_storage=read_storage[0],
                resize_storage=resize_storage[0],
                frame_size=frame_size,
            )
        else:
            future_objs = []
            for i in range(nc_num):
                obj = _imread_executor_pool.submit(
                    read_nc_resize,
                    path_list[i],
                    read_storage[i],
                    resize_storage[i],
                    frame_size,
                )
                future_objs.append(obj)
            wait(future_objs)

        resize_storage = resize_storage.reshape(
            (nc_num, 1, frame_size[0], frame_size[1])
        )

        return resize_storage[:, ::-1, ...]
    else:
        if nc_num == 1:
            read_nc(path=path_list[0], read_storage=read_storage[0])
        else:
            future_objs = []
            for i in range(nc_num):
                obj = _imread_executor_pool.submit(
                    read_nc, path_list[i], read_storage[i]
                )
                future_objs.append(obj)
            wait(future_objs)
        read_storage = read_storage.reshape((nc_num, 1, im_h, im_w))

        return read_storage[:, ::-1, ...]
