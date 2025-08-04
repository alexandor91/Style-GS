source /mnt/bd/dev-lantian/miniconda3/bin/activate
conda activate scaffold_gs

source_dir="/mnt/bd/dev-lantian/datasets_converted/cambridge/KingsCollege/train"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline/cambridge/KingsCollege"
lod=0
iterations=30_000
appearance_dim=64
update_init_factor=16
vsize=0.01
ratio=1
# resolution=1 # 这一项不启用的话，不会用4k的图像
warmup="True"

# cmd
cmd="python train.py \
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
