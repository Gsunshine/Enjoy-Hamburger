imagenet_dir=/home/imagenet
target_dir=/home/user_name/data/

mkdir $target_dir/tar_cache

num=1
for sub_class in $(ls ${imagenet_dir}/train/)
do
    echo $num
    echo $sub_class
    tar -zcPf ${target_dir}/tar_cache/${sub_class}.tar ${imagenet_dir}/train/${sub_class}
    let num++
done

tar -zcvPf ${target_dir}/ILSVRC2012_img_train.tar ${target_dir}/tar_cache/*
tar -zcvPf ${target_dir}/ILSVRC2012_img_val.tar ${imagenet_dir}/val
