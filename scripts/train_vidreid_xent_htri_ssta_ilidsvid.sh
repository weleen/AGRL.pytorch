#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  --seq-len 4 \
                                  --train-batch 64 \
                                  --test-batch 64 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --train-sampler RandomIdentitySampler \
                                  --optim amsgrad \
                                  --max-epoch 200 \
                                  --lr 3e-4 \
                                  --stepsize 100 \
                                  -a simple_sta \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 0 \
                                  --eval-step 1 \
                                  --split-id $i \
                                  --save-dir log/video/simple_sta/ilidsvid/split"$i"
let i=$i+1
done
