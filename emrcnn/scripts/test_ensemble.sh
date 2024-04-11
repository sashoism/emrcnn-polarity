# test immu
# 4 detectors will be used for testing

# If you have 4 GPUs and want to run all detectors parallely
for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$(($i)) python test_ensemble.py --data_name immu_ensemble --model_dir immu_ensemble --ensembleId $(($i+1)) --label_name immu&
done

# If you have a single GPU and want to run detectors one by one
# for i in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --data_name immu_ensemble --model_dir immu_ensemble --ensembleId $(($i+1)) --label_name immu
# done