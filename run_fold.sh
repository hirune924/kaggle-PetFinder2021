model_name=$1
batch_size=$2
image_size=$3
epoch=$4
agb=$5
group=$6

#model_name=tf_efficientnet_b0_ns
#batch_size=128
#epoch=100
#group=exp001-tf_efficientnet_b0_ns

for i in {0..4} ; do
    python exp001.py batch_size=${batch_size} epoch=${epoch} data_dir=../data/ output_dir=../output/ model_name=${model_name} fold=${i} group=${group} height=${image_size} width=${image_size} trainer.accumulate_grad_batches=${agb}
done

python oof001.py  batch_size=${batch_size} data_dir=../data/ output_dir=../output/ model_name=${model_name} group=${group} model_dir=../output height=${image_size} width=${image_size}
