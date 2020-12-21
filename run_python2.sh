#Cartesian product relations
#
echo results of Cartesian product relations;
cd ./Cartesian-product
mkdir output
python fb15k_cartesian_product.py > ./output/cartesian_product_results.out

# results of AMIE
#
echo results of AMIE;
cd ../AMIE
mkdir output
python combine_results.py > ./output/amie_results.out

#detailed results
#
echo detailed results;
cd ../Models-detailed-results
mkdir output
python result_by_relcategory.py > ./output/tables_9_10_12.out
python reverse_percentage.py > ./output/table_7.out
python detailed_results.py > ./output/table_8.out
python heatmap.py

#detailed results
#
echo simple rule model;
cd ../simple_rule
mkdir output
python simplerule_model.py > ./output/simple_rule.out

cd ../OpenKE_FB15k
bash make.sh
mkdir output
mkdir res
echo results of models using OpenKE framework;
echo "Training and testing models on FB15k";
#FB15k
python train_distmult.py > ./output/distmult_FB15K.out
echo results of DistMult are ready;

python train_complex.py > ./output/complex_FB15K.out
echo results of ComplEX are ready;

python train_transd.py > ./output/transd_FB15K.out
echo results of TransD are ready;

python train_transe.py > ./output/transe_FB15K.out
echo results of TransE are ready;

python train_transh.py > ./output/transh_FB15K.out
echo results of TransH are ready;

python train_transr.py > ./output/transr_FB15K.out
echo results of TransR are ready;

echo results of OpenKE on FB15k are saved in folder output;