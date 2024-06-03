# CPI-Parser

Official implementation of **CPI-Parser**. It based on [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Installation
- pytorch 1.10.0 
- python 3.7.0
- [mmdet 2.25.1](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

## Dataset
You need to download the datasets and annotations follwing this repo's formate


Make sure to put the files as the following structure:

```
  ├─data
  │  CIHP
  │  │  ├─train_img
  │  │  ├─train_parsing
  │  │  ├─train_seg
  │  │  ├─val_img
  │  │  ├─val_parsing
  │  │  ├─val_seg
  │  │  │─annotations
  │  MHP-v2
  │  │  ├─train_img
  │  │  ├─train_parsing
  │  │  ├─train_seg
  │  │  ├─val_img
  │  │  ├─val_parsing
  │  │  ├─val_seg
  │  │  │─annotations
  |
  ├─work_dirs
  |  ├─causal_parser_r50_inter_8_135k_cihp
  |  |  ├─epoch_75.pth
  ```

## Results

### CIHP

|  Backbone    |  LR  | mIOU | APvol | AP_p50 | PCP50 | download |
|--------------|:----:|:----:|:-----:|:------:|:-----:|:--------:|
|  R-50        |  3x  | 59.0 | 58.4  |  72.2  |  67.8 |[model](https://drive.google.com/file/d/1uiqPVZNWIEIKquMNhZ7tv3OTZrc9WGt0/view?usp=share_link) |
|  ConvNext-B  |  3x  | 66.5 | 63.1  |  80.9  |  74.6 |[model](https://drive.google.com/file/d/117grkZS6Q-v5tbeTLdyENSI0fSRD09lK/view?usp=sharing) |

### MHP

|  Backbone    |  LR  | mIOU | APvol | AP_p50 | PCP50 | download |
|--------------|:----:|:----:|:-----:|:------:|:-----:|:--------:|
|  R-50        |  3x  | 38.1 | 45.8  |  39.0  |  49.9 |[model](https://drive.google.com/file/d/15yp_c5ZNFbvkFn5K3wVkHix7F-m-3FBZ/view?usp=sharing) |
|  ConvNext-B  |  3x  | 41.8 | 48.9  |  48.6  |  57.4 |[model](https://drive.google.com/file/d/1qpdS8dEdhdjcYUOiDEiTusIyjifBAYCy/view?usp=sharing) |

## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/COSParser/causal_parser_r50_inter_8_135k_cihp.py work_dirs/causal_parser_r50_inter_8_135k_cihp/epoch75.pth 8 --eval bbox --eval-options "jsonfile_prefix=work_dirs/causal_parser_r50_inter_8_135k_cihp/causal_parser_r50_inter_8_135k_cihp_val_result"

# eval, noted that should change the json path produce by previous step.
python utils/eval.py
```

## Training
```
# training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/COSParser/causal_parser_r50_inter_8_135k_cihp.py 8
```