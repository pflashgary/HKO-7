import os
import pandas as pd
import itertools


idx = []
lst = ["./2020/%02d" % (i + 1) for i in range(0, 12)]
lst.append("./2021/01")
for i in lst:
    idx.append([x[0].split(".")[-1] for x in os.walk(i)])

pd.to_datetime(list(itertools.chain(*idx))).unique().dropna().to_frame().to_pickle(
    "/home/pfaeghlashgary/HKO-7/hko_data/pd/sst_train.pkl"
)

idx = [x[0].split(".")[-1] for x in os.walk("./2021/02")]
pd.to_datetime(idx).unique().dropna().to_frame().to_pickle(
    "/home/pfaeghlashgary/HKO-7/hko_data/pd/sst_test.pkl"
)

idx = [x[0].split(".")[-1] for x in os.walk("./2021/03")]
pd.to_datetime(idx).unique().dropna().to_frame().to_pickle(
    "/home/pfaeghlashgary/HKO-7/hko_data/pd/sst_valid.pkl"
)
