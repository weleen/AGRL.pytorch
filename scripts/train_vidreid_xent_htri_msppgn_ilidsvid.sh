#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  -a msppgn \
                                  --optim amsgrad \
                                  --lr 0.0003 \
                                  --height 256 \
                                  --width 128 \
                                  --seq-len 4 \
                                  --max-epoch 200 \
                                  --stepsize 200 \
                                  --margin 0.3 \
                                  --num-instances 8 \
                                  --train-sample restricted \
                                  --train-batch 72 \
                                  --test-batch 1 \
                                  --test-sample skipdense \
                                  --label-smooth \
                                  --split-id $i \
                                  --save-dir log/msppgn/ilidsvid-bs72-ni8-mg0.3-lr3e-4+np4+ap+af+pa+atp+zw70/split"$i" \
                                  --gpu-devices 0,1 \
                                  --print-freq 50 \
                                  --print-last \
                                  --eval-step 5 \
                                  --flip-aug \
                                  --use-pose \
                                  --learn-graph \
                                  --pose-aug \
                                  --pose-aggregate \
                                  --zero-wd 70
let i=$i+1
done