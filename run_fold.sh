model_name=$1
batch_size=$2
image_size=$3
epoch=$4
agb=$5
group=$6
#seed=$7

#model_name=tf_efficientnet_b0_ns
#batch_size=128
#epoch=100
#group=exp001-tf_efficientnet_b0_ns

for i in {0..4} ; do
    python exp009.py batch_size=${batch_size} epoch=${epoch} data_dir=../data/ output_dir=../output/ model_name=${model_name} fold=${i} group=${group} height=${image_size} width=${image_size} trainer.accumulate_grad_batches=${agb}
done

python oof009.py batch_size=${batch_size} data_dir=../data/ output_dir=../output/ model_name=${model_name} group=${group} model_dir=../output height=${image_size} width=${image_size}


#for i in {0..4} ; do
#    python exp008_nothink.py seed=1${seed} batch_size=${batch_size} epoch=${epoch} data_dir=../data/ output_dir=../output/ model_name=${model_name} fold=${i} group=${group}_center_${seed} height=${image_size} width=${image_size} trainer.accumulate_grad_batches=${agb}
#done

#python oof008_nothink.py seed=1${seed} batch_size=${batch_size} data_dir=../data/ output_dir=../output/ model_name=${model_name} group=${group}_center_${seed} model_dir=../output height=${image_size} width=${image_size}


#for i in {0..4} ; do
#    python exp008_nothink_1.py seed=2${seed} batch_size=${batch_size} epoch=${epoch} data_dir=../data/ output_dir=../output/ model_name=${model_name} fold=${i} group=${group}_resize_${seed} height=${image_size} width=${image_size} trainer.accumulate_grad_batches=${agb}
#done

#python oof008_nothink_1.py seed=2${seed} batch_size=${batch_size} data_dir=../data/ output_dir=../output/ model_name=${model_name} group=${group}_resize_${seed} model_dir=../output height=${image_size} width=${image_size}


#for i in {0..4} ; do
#    python exp008_nothink_2.py seed=1${seed} batch_size=${batch_size} epoch=${epoch} data_dir=../data/ output_dir=../output/ model_name=${model_name} fold=${i} group=${group}_center_kf_${seed} height=${image_size} width=${image_size} trainer.accumulate_grad_batches=${agb}
#done

#python oof008_nothink_2.py seed=1${seed} batch_size=${batch_size} data_dir=../data/ output_dir=../output/ model_name=${model_name} group=${group}_center_kf_${seed} model_dir=../output height=${image_size} width=${image_size}
