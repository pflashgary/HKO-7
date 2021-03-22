from netCDF4 import Dataset, num2date
sst_path = './crop.nc'

valid_time = []
with Dataset(sst_path) as cur_nc:
    sst = cur_nc.variables['analysed_sst'][:]
    time = cur_nc.variables['time'][:]
    calendar = 'gregorian'
    units = 'seconds since 1981-01-01 00:00:00 UTC'
    for t in range(len(time)):
        valid_time.append(num2date(
            time[t], units=units,
            calendar=calendar))


import matplotlib.pyplot as plt
import numpy as np

sst = sst.reshape(sst.shape[-2],sst.shape[-1])[750:750+480, 500:500+480]
print(sst.shape)


plt.imshow(sst, cmap='rainbow')
        

