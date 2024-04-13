#!/bin/bash
source /venv/bin/activate

python dataset/createPolygons.py --data_name $DATA_NAME
if [[ -z "$PRETRAINED" ]]; then
  for ((id=1; id<=ENSEMBLES; id++)); do
    python train_ensemble.py --data_name $DATA_NAME --ensembleId $id
  done
fi

for ((id=1; id<=ENSEMBLES; id++)); do
  python test_ensemble.py --data_name $DATA_NAME --ensembleId $id
done
python ensemble_fusion.py --data_name $DATA_NAME
python slice_merge.py --data_name $DATA_NAME