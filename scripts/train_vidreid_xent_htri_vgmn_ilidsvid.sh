#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  --seq-len 8 \
                                  --train-batch 32 \
                                  --test-batch 32 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --optim adam \
                                  --soft-margin \
                                  --max-epoch 400 \
                                  --lr 1e-4 \
                                  --stepsize 100 200 300 \
                                  -a vmgn \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 3 \
                                  --use-pose \
                                  --learn-graph \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 0,1 \
                                  --eval-step 1 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --save-dir log/video/vmgn/ilidsvid-ngb3-consistent/split"$i"
let i=$i+1
done
