#!/bin/bash
mkdir data
mkdir saved_models
cp -r ../datasets/WN18RR ./data
cp -r ../datasets/YAGO3-10 ./data
cp -r ../datasets/FB15k-237 ./data
cp -r ../datasets/FB15k ./data
cp -r ../datasets/WN18 ./data
cp -r ../datasets/YAGO3-10-DR ./data

python wrangle_KG.py WN18RR
python wrangle_KG.py WN18
python wrangle_KG.py YAGO3-10
python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k
python wrangle_KG.py YAGO3-10-DR

