# inference for a large volume using divide-and-conquer strategy

# inference immu as an example
CUDA_VISIBLE_DEVICES=0 python inference_divide-and-conquer.py \
                    --data_name immu_large_ensemble --model_dir immu_ensemble \
                    --test_img_dir /data/wu1114/Documents/dataset/Detectron/immu/test2/syn \
                    --k_max 200 --M 4
