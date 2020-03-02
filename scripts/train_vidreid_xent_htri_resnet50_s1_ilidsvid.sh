#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  --seq-len 8 \
                                  --train-batch 16 \
                                  --test-batch 16 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --train-sampler RandomIdentitySamplerV1 \
                                  --optim adam \
                                  --soft-margin \
                                  --max-epoch 200 \
                                  --lr 1e-4 \
                                  --stepsize 50 100 150 \
                                  -a resnet50_s1 \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 0 \
                                  --eval-step 1 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --save-dir log/video/resnet50_s1/ilidsvid/split"$i"
let i=$i+1
done
