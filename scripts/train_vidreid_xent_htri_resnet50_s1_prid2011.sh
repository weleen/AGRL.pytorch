#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d prid2011 \
                                  --seq-len 8 \
                                  --train-batch 16 \
                                  --test-batch 16 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --optim adam \
                                  --margin 0.3 \
                                  --max-epoch 200 \
                                  --lr 1e-4 \
                                  --stepsize 50 100 150 \
                                  -a resnet50_s1 \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 1 \
                                  --eval-step 1 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --save-dir log/video/resnet50_s1/prid2011/split"$i"
let i=$i+1
done
