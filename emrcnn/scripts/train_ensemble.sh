# training script
# 4 detectors will be trained for this demo

# If you have 4 GPUs and want to train all detectors parallely
# for i in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=$(($i)) python train_ensemble.py --ensembleId $(($i+1)) --data_name immu_ensemble&
# done


# If you have a single GPU and want to train detectors one by one
for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python train_ensemble.py --ensembleId $(($i+1)) --data_name immu_ensemble
done