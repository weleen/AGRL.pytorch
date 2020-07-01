## Introduction
This is the implementation for **Adaptive Graph Representation Learning for Video Person Re-identification**

## Get started
1. `cd` to the folder where you want to download this repo.
2. Run `git clone this_project`.
3. Install dependencies by `pip install -r requirements.txt`.
4. To accelerate evaluation (10x faster), you can use cython-based evaluation code (developed by [luzai](https://github.com/luzai)). First `cd` to `eval_lib`, then do `make` or `python setup.py build_ext -i`. After that, run `python test_cython_eval.py` to test if the package is successfully installed.

## Train
Training codes are implemented in
* `train_vidreid_xent_htri.py`: train video model with combination of cross entropy loss and hard triplet loss.

For example, to train an image reid model using ResNet50 and cross entropy loss, run
```bash
python train_vidreid_xent_htri.py -d mars -a vmgn --optim adam --lr 0.0003 --max-epoch 60 --stepsize 20 40 --train-batch 32 --test-batch 100 --save-dir log/resnet50-xent-market1501 --gpu-devices 0
```

To use multiple GPUs, you can set `--gpu-devices 0,1,2,3`.

**Note:** To resume training, you can use `--resume path/to/.pth.tar` to load a checkpoint from which saved model weights and `start_epoch` will be used. Learning rate needs to be initialized carefully. If you just wanna load a pretrained model by discarding layers that do not match in size (e.g. classification layer), use `--load-weights path/to/.pth.tar` instead.

Please refer to the code for more details.


## Test
create a directory to store model weights `mkdir saved-models/` beforehand. Then, run the following command to test
```bash
python train_vidreid_xent_htri.py -d mars -a vmgn --evaluate --resume saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --test-batch 100 --gpu-devices 0
```

**Note** that `--test-batch` in video reid represents number of tracklets. If you set this argument to 2, and sample 15 images per tracklet, the resulting number of images per batch is 2*15=30. Adjust this argument according to your GPU memory.

## Reference
This project is heavily related to [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [STE-NVAN](https://github.com/jackie840129/STE-NVAN/).
