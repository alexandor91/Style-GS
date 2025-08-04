source /mnt/bd/dev-lantian/miniconda3/bin/activate
conda activate scaffold_gs

lod=0
iterations=1_000
appearance_dim=0
update_init_factor=16
vsize=0.01
ratio=1
resolution=1 # 这一项不启用的话，不会用4k的图像
warmup="True"
source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Truck"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Truck"
stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result/Tanks-and-Temples/Truck/38"
style_img="images/examples/38.jpg"
cmd="python transfer_all.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --resolution $resolution \
    --ratio $ratio \
    --style_img $style_img \
    --output_path $stylized_dir "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result/Tanks-and-Temples/Truck/42"
style_img="images/examples/42.jpg"
cmd="python transfer_all.py \
    --eval \
    -s $source_dir \
    -m $output_dir \
    --iterations $iterations \
    --lod $lod \
    --voxel_size $vsize \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --resolution $resolution \
    --ratio $ratio \
    --style_img $style_img \
    --output_path $stylized_dir "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan106"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline/DTU/scan106"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/dgso_r/41/DTU/scan106"
# style_img="images/41.jpg"
# cmd="python transfer.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --ratio $ratio \
#     --style_img $style_img \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s