#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d mars \
                                  -a msppgn \
                                  --optim amsgrad \
                                  --lr 0.0003 \
                                  --height 256 \
                                  --width 128 \
                                  --seq-len 4 \
                                  --max-epoch 600 \
                                  --stepsize 200 400 \
                                  --margin 0.3 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --train-batch 72 \
                                  --test-batch 1 \
                                  --test-sample skipdense \
                                  --label-smooth \
                                  --split-id $i \
                                  --save-dir log/msppgn/mars-bs72-ni4-mg0.3-lr3e-4+ap+af+pa+atp+zw70/split"$i" \
                                  --gpu-devices 0,1 \
                                  --print-freq 50 \
                                  --print-last \
                                  --eval-step 20 \
                                  --flip-aug \
                                  --use-pose \
                                  --learn-graph \
                                  --pose-aug \
                                  --pose-aggregate
let i=$i+1
done