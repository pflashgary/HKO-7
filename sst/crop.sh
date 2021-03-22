#! /bin/bash
for file in $(find . -name '*.nc'); do
    ncks -d lat,-50.,-30. -d lon,165.,180. ${file}  -O  "${file%.*}_crop.nc"
done