# AICUP Baseline: BoT-SORT

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
> 
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bot-sort-robust-associations-multi-pedestrian/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bot-sort-robust-associations-multi-pedestrian)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bot-sort-robust-associations-multi-pedestrian/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bot-sort-robust-associations-multi-pedestrian)

> [!IMPORTANT]  
> **This baseline is based on the code released by the original author of [BoT-SORT](https://github.com/NirAharon/BoT-SORT). Special thanks for their release.**


> [!WARNING]
>  - **This baseline only provides single-camera object tracking and does not include cross-camera association.**
>  - **Due to our dataset's low frame rate (fps: 1), we have disabled the Kalman filter in BoT-SORT. Low frame rates can cause the Kalman filter to deviate, hence we only used appearance features for tracking in this baseline.**


## ToDo
- [ ] Fine-tune `yolov7`
- [ ] Implement The output of the `demo code`
- [ ] Upload baseline score
- [ ] Release test set

### Visualization results on AICUP train_set


## Installation

The code was tested on Ubuntu 20.04

BoT-SORT code is based on ByteTrack and FastReID. <br>
Visit their installation guides for more setup options.
 
### Setup with Conda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n botsort python=3.7
conda activate botsort
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 1.11.0+cu113 and torchvision==0.12.0 

**Step 3.** Fork this Repository and clone your Repository to your device

**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step 5. Others
```shell
# Cython-bbox
pip3 install cython_bbox

# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

## Data Preparation

Download the AI_CUP dataset, original dataset structure is:
```python
├── train
│   ├── images
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── ...
│   └── labels
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.txt (CamID_FrameNum)
│   │   │  ├── 0_00002.txt
│   │   │  ├── ...
│   │   │  ├── 1_00001.txt (CamID_FrameNum)
│   │   │  ├── 1_00002.txt
│   │   │  ├── ...
│   │   │  ├── 7_00001.txt (CamID_FrameNum)
│   │   │  ├── 7_00002.txt
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.txt (CamID_FrameNum)
│   │   │  ├── 0_00002.txt
│   │   │  ├── ...
│   │   │  ├── 1_00001.txt (CamID_FrameNum)
│   │   │  ├── 1_00002.txt
│   │   │  ├── ...
│   │   │  ├── 7_00001.txt (CamID_FrameNum)
│   │   │  ├── 7_00002.txt
│   │   ├── ...
--------------------------------------------------
├── test
│   ├── images
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── ...
```

For training the ReID, detection patches must be generated as follows:   

```shell
cd <BoT-SORT_dir>

# For AICUP 
python fast_reid/datasets/generate_AICUP_patches.py --data_path <dataets_dir>
```

> [!TIP]
> You can link dataset to FastReID ```export FASTREID_DATASETS=<BoT-SORT_dir>/fast_reid/datasets```. If left unset, the default is `fast_reid/datasets` 

 
## Model Zoo for MOT17 & COCO
Download and store the trained models in 'pretrained' folder as follow:
```
<BoT-SORT_dir>/pretrained
```
- We used the publicly available [ByteTrack](https://github.com/ifzhang/ByteTrack) model zoo trained on MOT17, MOT20 and ablation study for YOLOX object detection.

- Author's trained ReID models can be downloaded from [MOT17-SBS-S50](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view?usp=sharing).

- For multi-class MOT use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) or [YOLOv7](https://github.com/WongKinYiu/yolov7) trained on COCO (or any custom weights). 

## Training

### Train the ReID Module for AICUP

After generating AICUP ReID dataset as described in the 'Data Preparation' section.

```shell
cd <BoT-SORT_dir>

# For training AICUP 
python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```

The training results are stored by default in ```logs/AICUP/bagtricks_R50-ibn```. The storage location and model hyperparameters can be modified in ```fast_reid/configs/AICUP/bagtricks_R50-ibn.yml```.

Refer to [FastReID](https://github.com/JDAI-CV/fast-reid)  repository for addition explanations and options.

> [!IMPORTANT]  
> Since we did not generate the `query` and `gallery` datasets required for evaluation when producing the ReID dataset (`MOT17_ReID` provided by BoT-SORT also not provide them), please skip the following TrackBack when encountered after training completion.

```shell
Traceback (most recent call last):
...
File "./fast_reid/fastreid/evaluation/reid_evaluation.py", line 107, in evaluate
    cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
  File "./fast_reid/fastreid/evaluation/rank.py", line 198, in evaluate_rank
    return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
  File "rank_cy.pyx", line 20, in rank_cy.evaluate_cy
  File "rank_cy.pyx", line 28, in rank_cy.evaluate_cy
  File "rank_cy.pyx", line 240, in rank_cy.eval_market1501_cy
AssertionError: Error: all query identities do not appear in gallery
```

## Tracking (Demo)
> [!TIP]
> We recommend using YOLOv7 as the object detection model for tracking

Track AICUP with BoT-SORT(-ReID) based YOLOv7 and multi-class.
```shell
cd <BoT-SORT_dir>
python3 tools/mc_demo_yolov7.py --weights pretrained/yolov7-e6e.pt --source AI_CUP_MCMOT_dataset/train/images/0903_150000_151900 --fuse-score --agnostic-nms --with-reid --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP/bagtricks_R50-ibn/model_0058.pth
```


Demo with BoT-SORT(-ReID) based YOLOX and multi-class.
```shell
cd <BoT-SORT_dir>

# Original example
python3 tools/demo.py video --path <path_to_video> -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result

# Multi-class example
python3 tools/mc_demo.py video --path <path_to_video> -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
```

## Note

Our camera motion compensation module is based on the OpenCV contrib C++ version of VideoStab Global Motion Estimation, 
which currently does not have a Python version. <br>
Motion files can be generated using the C++ project called 'VideoCameraCorrection' in the GMC folder. <br> 
The generated files can be used from the tracker. <br>

In addition, python-based motion estimation techniques are available and can be chosen by passing <br> 
'--cmc-method' <files | orb | ecc> to demo.py or track.py. 

## Citation

```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```


## Acknowledgement

A large part of the codes, ideas and results are borrowed from
[BoT-SORT](https://github.com/NirAharon/BoT-SORT) 
[ByteTrack](https://github.com/ifzhang/ByteTrack), 
[StrongSORT](https://github.com/dyhBUPT/StrongSORT),
[FastReID](https://github.com/JDAI-CV/fast-reid),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[YOLOv7](https://github.com/wongkinyiu/yolov7). 
Thanks for their excellent work!











