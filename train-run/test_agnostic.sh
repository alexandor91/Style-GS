source /mnt/bd/dev-lantian/miniconda3/bin/activate
conda activate scaffold_gs

lod=0
iterations=3_000
appearance_dim=0
update_init_factor=16
vsize=0.01
ratio=1
resolution=1 # 这一项不启用的话，不会用4k的图像
warmup="True"
style_img="images/examples/38.jpg"
style_path="images/examples/"

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Truck"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Truck"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/Truck/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Train"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Train"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/Train/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Family"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Family"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/Family/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Horse"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Horse"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/Horse/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/Caterpillar"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/Caterpillar"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/Caterpillar/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/TNT/M60"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/Tanks-and-Temples/M60"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/Tanks-and-Temples/M60/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan106"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/DTU/scan106"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/DTU/scan106/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan24"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/DTU/scan24"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/DTU/scan24/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

# source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan122"
# output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/DTU/scan122"
# stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/DTU/scan122/"
# cmd="python transfer_all_agnostic.py \
#     --eval \
#     -s $source_dir \
#     -m $output_dir \
#     --iterations $iterations \
#     --lod $lod \
#     --voxel_size $vsize \
#     --update_init_factor $update_init_factor \
#     --appearance_dim $appearance_dim \
#     --resolution $resolution \
#     --ratio $ratio \
#     --style_img $style_img \
#     --style_path $style_path \
#     --output_path $stylized_dir "
# if [ "$warmup" = "True" ]; then
#     cmd="$cmd --warmup"
# fi
# eval $cmd
# sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan65"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/DTU/scan65"
stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/DTU/scan65/"
cmd="python transfer_all_agnostic.py \
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
    --style_path $style_path \
    --output_path $stylized_dir "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s

source_dir="/mnt/bd/dev-lantian/datasets_converted/DTU/scan114"
output_dir="/mnt/bd/dev-lantian/exp_results/scgs/baseline_sh_full_res_no_app/DTU/scan114"
stylized_dir="/mnt/bd/dev-lantian/exp_results/final_result_agnostic/DTU/scan114/"
cmd="python transfer_all_agnostic.py \
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
    --style_path $style_path \
    --output_path $stylized_dir "
if [ "$warmup" = "True" ]; then
    cmd="$cmd --warmup"
fi
eval $cmd
sleep 20s