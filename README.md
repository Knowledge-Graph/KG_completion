# Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study
This repository contains codes, experiment scripts, and datasets to reproduce results of the following papers:

[Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study](https://arxiv.org/abs/2003.08001) Farahnaz Akrami, Mohammed Samiul Saeef, Qingheng Zhang, Wei Hu, and Chengkai Li, SIGMOD 2020

[Re-evaluating Embedding-Based Knowledge Graph Completion Methods](https://dl.acm.org/citation.cfm?id=3269266) Farahnaz Akrami, Lingbing Guo, Wei Hu, and Chengkai Li, CIKM 2018

A summary of the paper is published in a [Medium blog post](https://link.medium.com/lBHwjLeI94).

## Results of different models on various datasets
There are a total of 9 embedding-based models TransE, TransH, TransR, TransD, DistMult, ComplEx, ConvE, RotatE, and TuckER that should be trained on 4 different datasets FB15K, FB15K-237, WN18, WN18RR to obtain the results of Table 5, 6, and 13 of the paper.  

Furthermore, we trained TransE, DistMult, ComplEX, ConvE, RotatE, TuckER on two datasets YAGO3-10 and YAGO3-10-DR to have the results of Table 11.

The experiments used source codes of various methods from several places, including the OpenKE repository which covers implementations of TransE, TransH, TransR, TransD, DistMult, and ComplEx, as well as the source code releases of ComplEx (which also covers DistMult), ConvE, RotatE, and TuckER. 

The commands to train and test different models can be found in shell script *run_python3.sh*. The results of OpenKE on FB15k can be obtained by running the shell script named *run_python2.sh*.

After the training and test are completed for a model, the results of that model will be saved in a folder named **_output_** located inside the folder that has the implementation of that model. For example, you can find the results of **TransE** on **FB15k** obtained by using the **OpenKE** framework in this path **_./KG_completion/OpenKE/output/transe_FB15k.out_**


### ComplEx
* Cloned October 2018.
* Implementations for ComplEx and DistMult.
* Original repository: (https://github.com/ttrouill/complex).
* evaluation.py is modified to calculate raw metrics.

### ConvE
* Cloned October 2020
* Implementations for ConvE.
* Original repository: (https://github.com/TimDettmers/ConvE).
* evaluation.py is modified to print results for each test triple and calculate raw metrics.

### OpenKE
* Cloned October 2019.
* Implementations for TransE.
* Original repository: (https://github.com/thunlp/OpenKE).
* Test.h is modified to print results for each test triple.

### RotatE
* Cloned October 2019.
* Implementations for RotatE.
* Original repository: (https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
* Modified to print results for each test triple.

### TuckER
* Cloned October 2019.
* Implementations for TuckER.
* Original repository: (https://github.com/ibalazevic/TuckER).
* Modified to print results for each test triple.

## Cartesian-product
To find Cartesian Product relations in FB15k and perform link prediction on these relations:
```
python fb15k_cartesian_product.py
```

## AMIE
Implementations for conducting link prediction using rules generated by AMIE.
### Run the experiments
Head and tail entity link prediction for a specific test relation:

```
python left.py relation_id
python right.py relation_id
```
Results obtained for each test relation are stored in ./AMIE_LinkPrediction_results. 

To have the results of link prediction using AMIE:
```
python combine_res.py
```

## Models-detailed-results
To obtain these results, the results for each test triple obtained by training and testing different models should be stored in directory *./KG_completion/Models-detailed-results/test_results*

Link prediction results for each test relation of FB15k-237, WN18RR, and YAGO3-10:
```
python result_by_rel_all.py
```

To calculate the head and tail entity link prediction results for different relation types:
```
python result_by_relcategory.py
```

Percentage of triples on which each method outperforms others, separately for each relation:

```
python heatmap.py
```

Percentage of test triples with better performance than TransE that have reverse and duplicate triples in training set:

```
python reverse_percentage.py
```

To obtain the number of relations on which each model is the most accurate and charts:

```
python detailed_results.py
```

## Data redundancy
To find the reverse and duplicate relations in FB15k:
```
python FB15k_redundancy.py
```
To find redundant information in test set of FB15k:
```
python FB15k_test_redundancy.py
```



