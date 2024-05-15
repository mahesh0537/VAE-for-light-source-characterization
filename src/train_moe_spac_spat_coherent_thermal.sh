nbar=1.9
quantum_efficiency=0.6
n_sensor=4
nepochs=3000
out_dir=./checkpoints_multi
max_photons=20
involve_loss=True
data_path=../training_data_mix
name=MoE_spac_spat_C_T
use_moe=True
out_class=4



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
CUDA_VISIBLE_DEVICES=0, python3 train.py $flag
