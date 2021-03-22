About
-----

Source code of the paper [Deep learning for precipitation nowcasting: A benchmark and a new model](http://papers.nips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model)

If you use the code or find it helpful, please cite the following paper:
```
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
```

Installation
------------

**Requires Python 3.5 or newer!**

Both Windows and Linux are supported.

Install the package
```bash
python3 setup.py develop
# Use --user if you have no privilege
python3 setup.py develop --user
```

You will also need the python plugin of opencv:
```bash
pip3 install opencv-contrib-python
```

In addition, you will need to install FFMpeg + X264 (See FAQ).

For windows users it may be difficult to install some required packages like
numba, ffmpeg or opencv-python. We strongly recommend you to use
[Anaconda](https://www.anaconda.com/download/) and install them by commands like
`conda install numba`. To install opencv-python on windows, you can download the
wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/.

If you want to run the deep models in the paper, e.g., TrajGRU, you will need to install [MXNet](https://github.com/apache/incubator-mxnet). We've tested our code under [MXNet v0.12.0](https://github.com/apache/incubator-mxnet/releases/tag/0.12.0).
Also, in order to run the ROVER algorithm, install the python wrapper of VarFlow by following the guide in [VarFlow](https://github.com/sxjscience/HKO-7/tree/master/VarFlow).

**IMPORTANT!** You are able to run the HKO-7 benchmark environment without MXNet or VarFlow. You can proceed to use the HKOIterator and HKOBenchmarkEnv after you have installed the python package + Opencv-Python + FFMpeg with X264 encoding enabled and have downloaded the data. (See sections below for more reference).