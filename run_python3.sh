#!/bin/sh

#download repository
#git clone https://github.com/Knowledge-Graph/KG_completion  

#Training models using OpenKE
cd ./OpenKE
bash make.sh
mkdir output
mkdir res
echo results of models using OpenKE framework;
echo "Training and testing models on FB15k-237";
#FB15k-237
python train_distmult_FB15K237.py > ./output/distmult_FB15K237.out
echo results of DistMult are ready;

python train_complex_FB15K237.py > ./output/complex_FB15K237.out
echo results of ComplEX are ready;

python train_transd_FB15K237.py > ./output/transd_FB15K237.out
echo results of TransD are ready;

python train_transe_FB15K237.py > ./output/transe_FB15K237.out
echo results of TransE are ready;

python train_transh_FB15K237.py > ./output/transh_FB15K237.out
echo results of TransH are ready;

python train_transr_FB15K237.py > ./output/transr_FB15K237.out
echo results of TransR are ready;

echo results of OpenKE on FB15k-237 are saved in folder output;

#WN18
echo "Training and testing models on WN18";

python train_complex_WN18.py > ./output/complex_WN18.out
echo results of ComplEX are ready;

python train_distmult_WN18.py > ./output/distmult_WN18.out
echo results of DistMult are ready;

python train_transd_WN18.py > ./output/transd_WN18.out
echo results of TransD are ready;

python train_transe_WN18.py > ./output/transe_WN18.out
echo results of TransE are ready;

python train_transh_WN18.py > ./output/transh_WN18.out
echo results of TransH are ready;

python train_transr_WN18.py > ./output/transr_WN18.out
echo results of TransR are ready; 
echo results of OpenKE on WN18 are saved in folder output;

#WN18RR
echo "Training and testing models on WN18RR";

python train_complex_WN18RR.py > ./output/complex_WN18RR.out
echo results of ComplEX are ready;

python train_distmult_WN18RR.py > ./output/distmult_WN18RR.out
echo results of DistMult are ready;

python train_transd_WN18RR.py > ./output/transd_WN18RR.out &
echo results of TransD are ready;

nohup python train_transe_WN18RR.py > ./output/transe_WN18RR.out &
echo results of TransE are ready;

nohup python train_transh_WN18RR.py > ./output/transh_WN18RR.out &
echo results of TransH are ready;

nohup python train_transr_WN18RR.py > ./output/transr_WN18RR.out &
echo results of TransR are ready; 
echo results of OpenKE on WN18RR are saved in folder output;

#Training models using ConvE source code
#
echo results of models using ConvE source code;
cd ../ConvE
mkdir output
echo installing requirements;
pip install -r requirements.txt
wait
echo Download the default English model used by spaCy, which is installed in the previous step;
python -m spacy download en
wait
echo Run the preprocessing script for WN18, WN18RR, FB15k, FB15k-237, YAGO3-10, YAGO-3-10-DR
sh preprocess.sh
wait
echo Training ConvE model on FB15k-237, FB15k, WN18, WN18RR, YAGO3-10, YAGO-3-10-DR

CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.003 --preprocess > ./output/conve_FB15k237.out
									  
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/conve_FB15k.out
									  
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/conve_WN18.out

CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.003 --preprocess > ./output/conve_WN18RR.out
									  
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data YAGO3-10 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/conve_YAGO3-10.out

CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data YAGO3-10-DR \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/conve_YAGO3-10-DR.out									  


CUDA_VISIBLE_DEVICES=0 python main.py --model complex --data YAGO3-10 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/complex_YAGO3-10.out

CUDA_VISIBLE_DEVICES=0 python main.py --model complex --data YAGO3-10-DR \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/complex_YAGO3-10-DR.out
									  
CUDA_VISIBLE_DEVICES=0 python main.py --model distmult --data YAGO3-10 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/distmult_YAGO3-10-DR.out

CUDA_VISIBLE_DEVICES=0 python main.py --model distmult --data YAGO3-10-DR \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.001 --preprocess > ./output/distmult_YAGO3-10-DR.out	

echo results of ConvE are ready and stored in folder ConvE/output	

# Best Configuration for RotatE
#
echo results of RotatE;
cd ../RotatE
mkdir output
 
nohup bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de > ./output/rotate_FB15k.out &
wait
nohup bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de > ./output/rotate_FB15k237.out &
wait
nohup bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de > ./output/rotate_WN18.out &
wait
nohup bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de > ./output/rotate_WN18RR.out &
wait
nohup bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de > ./output/rotate_YAGO3-10.out &
wait
nohup bash run.sh train RotatE YAGO3-10-DR 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de > ./output/rotate_YAGO3-10-DR.out &
wait	
nohup bash run.sh train TransE YAGO3-10 3 0 1024 400 500 24.0 1.0 0.0002 100000 4 > ./output/transe_YAGO3-10.out &
wait
nohup bash run.sh train ComplEx YAGO3-10 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -de -dr -r 0.000002  > ./output/complex_YAGO3-10.out &
wait
nohup bash run.sh train DistMult YAGO3-10 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -r 0.000002 > ./output/distmult_YAGO3-10.out &
wait
nohup bash run.sh train TransE YAGO3-10-DR 3 0 1024 400 500 24.0 1.0 0.0002 100000 4 > ./output/transe_YAGO3-10-DR.out &
wait
nohup bash run.sh train ComplEx YAGO3-10-DR 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -de -dr -r 0.000002  > ./output/complex_YAGO3-10-DR.out &
wait
nohup bash run.sh train DistMult YAGO3-10-DR 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -r 0.000002 > ./output/distmult_YAGO3-10-DR.out &
wait
echo results of RotatE are ready in folder RotatE/output

# TuckER training
#
echo results of TuckER;
cd ../TuckER
mkdir output

CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k --num_iterations 500 --batch_size 128 --lr 0.003 --dr 0.99 \
                                      --edim 200 --rdim 200 --input_dropout 0.2 --hidden_dropout1 0.2 \
                                      --hidden_dropout2 0.3 --label_smoothing 0.0 > ./output/TuckER-FB15k.out

CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 \
                                      --edim 200 --rdim 200 --input_dropout 0.3 --hidden_dropout1 0.4 \
                                      --hidden_dropout2 0.5 --label_smoothing 0.1 > ./output/TuckER-FB15k-237.out

CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 0.995 \
                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.1 \
                                      --hidden_dropout2 0.2 --label_smoothing 0.1 > ./output/TuckER-WN18.out

CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18RR --num_iterations 500 --batch_size 128 --lr 0.01 --dr 1.0 \
                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 \
                                      --hidden_dropout2 0.3 --label_smoothing 0.1 > ./output/TuckER-WN18RR.out

CUDA_VISIBLE_DEVICES=0 python main.py --dataset YAGO3-10 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 1.0 \
                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 \
                                      --hidden_dropout2 0.3 --label_smoothing 0.1 > ./output/TuckER-YAGO3-10.out

CUDA_VISIBLE_DEVICES=0 python main.py --dataset YAGO3-10-DR --num_iterations 500 --batch_size 128 --lr 0.005 --dr 1.0 \
                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 \
                                      --hidden_dropout2 0.3 --label_smoothing 0.1 > ./output/TuckER-YAGO3-10-DR.out



echo results of ComplEX;
cd ../ComplEX
mkdir output
python wn18_run.py > ./output/complex_WN18.out
python fb15k_run.py > ./output/complex_FB15k.out












							  