nbar=1.9
quantum_efficiency=0.8
n_sensor=4
nepochs=3000
out_dir=./checkpoints_one
max_photons=20
involve_loss=True
data_path=../trainingData
name=spac_spat
use_moe=False
out_class=1


flag="--nbar $nbar
--quantum_efficiency $quantum_efficiency
--n_sensor $n_sensor
--epochs $nepochs
--out_dir $out_dir
--max_photons $max_photons
--involve_loss $involve_loss
--data_path $data_path
--name $name
--use_moe $use_moe
--out_class $out_class
"
echo $flag
CUDA_VISIBLE_DEVICES=4, python3 train.py $flag
