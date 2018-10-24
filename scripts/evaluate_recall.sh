#!/usr/bin/env bash 
cd third_party/densevid_eval
python evaluate.py  --tious 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -s $1
