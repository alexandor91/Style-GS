source /mnt/bd/dev-lantian/miniconda3/bin/activate
conda activate scaffold_gs

lod=0
iterations=30_000
appearance_dim=64
update_init_factor=16
vsize=0.01
ratio=1
# resolution=1 # 这一项不启用的话，不会用4k的图像
warmup="True"
### TNT ###
# source_dir="/mnt/bd/dev-lantian/datasets_converted/Tanks-and-Temples/Palace"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/Tanks-and-Temples/Palace"
# cmd="python train_jit.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/Tanks-and-Temples/Caterpillar"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/Tanks-and-Temples/Caterpillar"
# cmd="python train_jit.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/Tanks-and-Temples/Temple"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/Tanks-and-Temples/Temple"
# cmd="python train_jit.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/Tanks-and-Temples/Truck"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/Tanks-and-Temples/Truck"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

##### heritage recon #####
source_dir="/mnt/bd/dev-lantian/datasets_converted/heritage_recon/brandenburg_gate"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/heritage_recon/brandenburg_gate"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/heritage_recon/sacre_coeur"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/heritage_recon/sacre_coeur"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

# for A100-80g
# source_dir="/mnt/bd/dev-lantian/datasets_converted/heritage_recon/trevi_fountain"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/heritage_recon/trevi_fountain"
# cmd="python train_jit.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

### DTU Dataset ###
# source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan65"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/DTU/scan65"
# cmd="python train_jit.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan122"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/DTU/scan122"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan114"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/DTU/scan114"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan106"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/DTU/scan106"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan24"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_jit/DTU/scan24"
cmd="python train_jit.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --ratio $ratio "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s