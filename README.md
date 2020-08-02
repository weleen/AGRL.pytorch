## Introduction
This is the PyTorch implementation for **Adaptive Graph Representation Learning for Video Person Re-identification**.

## Get started
```shell
git clone https://github.com/weleen/AGRL.pytorch /path/to/save
pip install -r requirements.txt
cd torchreid/metrics/rank_cylib && make
```

## Dataset
Prepare datasets:
```shell
├── dukemtmc-vidreid
│   ├── DukeMTMC-VideoReID
│   ├── pose.json
│   ├── split_gallery.json
│   ├── split_query.json
│   └── split_train.json
│
├── ilids-vid
│   ├── i-LIDS-VID
│   ├── pose.json
│   ├── splits.json
│   └── train-test people splits
│
├── mars
│   ├── bbox_test
│   ├── bbox_train
│   ├── info
│   ├── pose.json
│   └── train-test people splits
│
├── prid
    ├── pose.json
    ├── prid_2011
    ├── prid_2011.zip
    ├── splits_prid2011.json
    └── train_test_splits_prid.mat
```
`pose.json` could be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1RduGEbq-tmfLAHM0k3xa4A) (code: luxr) and 
[Google Driver](https://drive.google.com/drive/folders/1BVEjMava3UQh4jC2bp-tcFo1rOZDB8MS?usp=sharing).

Details could be found in [DATASETS.md](DATASETS.md).


## Train
```bash
python train_vidreid_xent_htri.py -d mars -a vmgn --optim adam --lr 0.0003 --max-epoch 60 --stepsize 20 40 --train-batch 32 --test-batch 100 --save-dir log/resnet50-xent-market1501 --gpu-devices 0
```

To use multiple GPUs, you can set `--gpu-devices 0,1,2,3`.

**Note:** To resume training, you can use `--resume path/to/.pth.tar` to load a checkpoint from which saved model weights and `start_epoch` will be used. Learning rate needs to be initialized carefully. If you just wanna load a pretrained model by discarding layers that do not match in size (e.g. classification layer), use `--load-weights path/to/.pth.tar` instead.

Please refer to the code for more details.


## Test
create a directory to store model weights `mkdir saved-models/` beforehand. Then, run the following command to test
```bash
python train_vidreid_xent_htri.py -d mars -a vmgn --evaluate --resume saved-model/model.pth.tar --save-dir log --test-batch 100 --gpu-devices 0
```

## Citation
Please kindly cite this project in your paper if it is helpful😊:
```
@article{wu2020adaptive,
  title={Adaptive graph representation learning for video person re-identification},
  author={Wu, Yiming and Bourahla, Omar El Farouk and Li, Xi* and Wu, Fei and Tian, Qi and Zhou, Xue},
  journal={IEEE Transactions on Image Processing},
  year={2020},
  publisher={IEEE}
}
```

This project is developed based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [STE-NVAN](https://github.com/jackie840129/STE-NVAN/).
