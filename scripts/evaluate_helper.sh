#!/usr/bin/env bash 
cd third_party/densevid_eval
python evaluate.py -s $1 2>&1| cat >$2
