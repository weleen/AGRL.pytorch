#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d prid2011 \
                                  --seq-len 4 \
                                  --train-batch 64 \
                                  --test-batch 64 \
                                  --num-instances 8 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --train-sampler RandomIdentitySampler \
                                  --optim adam \
                                  --max-epoch 400 \
                                  --lr 3e-4 \
                                  --stepsize 200 \
                                  -a sta \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 4,5 \
                                  --eval-step 10 \
                                  --split-id $i \
                                  --save-dir log/video/sta/prid2011/split"$i"
let i=$i+1
done
