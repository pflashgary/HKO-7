#! /bin/bash

find . -name '*.bz2' | xargs bzip2 -d
