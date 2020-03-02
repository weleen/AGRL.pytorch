#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  --seq-len 8 \
                                  --train-batch 32 \
                                  --test-batch 1 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample skipdense \
                                  --optim adam \
                                  --margin 0.3 \
                                  --max-epoch 200 \
                                  --lr 3e-4 \
                                  --stepsize 100 150 \
                                  -a gsta \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 1 \
                                  --use-pose \
                                  --learn-graph \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 4,5 \
                                  --eval-step 5 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --save-dir log/video/gsta/ilidsvid-ngb1/split"$i"
let i=$i+1
done
