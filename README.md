
## Paper
Xiaohang Zhan, Xingang Pan, Bo Dai, Ziwei Liu, Dahua Lin, Chen Change Loy, "[Self-Supervised Scene-De-occlusion](https://arxiv.org/abs/2004.02788)", accepted to CVPR 2020 as oral. [[Project page](https://xiaohangzhan.github.io/projects/deocclusion/)].

For further information, please contact [Xiaohang Zhan](https://xiaohangzhan.github.io/).

## Demos

Watch the demo video in [YouTube](https://www.youtube.com/watch?v=xIHCyyaB5gU) or [bilibili](https://www.bilibili.com/video/BV1JT4y157Wt). The demo video contains vivid explanations of the idea, and interesting applications.

## Requirements

* pytorch>=0.4.1

    ```shell
    pip install -r requirements.txt
    ```

## Data prepration

### COCOA dataset proposed in [Semantic Amodal Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Semantic_Amodal_Segmentation_CVPR_2017_paper.pdf).

1. Download COCO2014 train and val images from [here](http://cocodataset.org/#download) and unzip.

2. Download COCOA annotations from [here](https://github.com/Wakeupbuddy/amodalAPI) and untar.

3. Ensure the COCOA folder looks like:

    ```
    COCOA/
      |-- train2014/
      |-- val2014/
      |-- annotations/
        |-- COCO_amodal_train2014.json
        |-- COCO_amodal_val2014.json
        |-- COCO_amodal_test2014.json
        |-- ...
    ```

4. Create symbolic link:
    ```
    cd deocclusion
    mkdir data
    cd data
    ln -s /path/to/COCOA
    ```

### KINS dataset proposed in [Amodal Instance Segmentation with KINS Dataset](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qi_Amodal_Instance_Segmentation_With_KINS_Dataset_CVPR_2019_paper.pdf).

1. Download left color images of object data in KITTI dataset from [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and unzip.

2. Download KINS annotations from [here](https://drive.google.com/drive/folders/1hxk3ncIIoii7hWjV1zPPfC0NMYGfWatr?usp=sharing) corresponding to [this commit](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset/tree/fb7be3fcedc96d4a6e20d4bb954010ec1b4f3194).

3. Ensure the KINS folder looks like:

    ```
    KINS/
      |-- training/image_2/
      |-- testing/image_2/
      |-- instances_train.json
      |-- instances_val.json
    ```

4. Create symbolic link:
    ```
    cd deocclusion/data
    ln -s /path/to/KINS
    ```

### [LVIS](https://www.lvisdataset.org/) dataset

1. Download training and validation sets from [here](https://www.lvisdataset.org/dataset)


## Run demos

1. Download released models [here](https://drive.google.com/drive/folders/1O89ItVWucCoL_VxIbLM1XLxr9JFfyj_Y?usp=sharing) and put the folder `released` under `deocclusion`.

2. Run `demos/demo_cocoa.ipynb` or `demos/demo_kins.ipynb`.

## Train

### train PCNet-M

1. Train (taking COCOA for example).

    ```
    sh experiments/COCOA/pcnet_m/train.sh
    ```

2. Monitoring status and visual results using tensorboard.

    ```
    sh tensorboard.sh $PORT
    ```

### train PCNet-C

1. Download the pre-trained image inpainting model using partial convolution [here](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/README.md) to `pretrains/partialconv.pth`

2. Convert the model to accept 4 channel inputs.

    ```shell
    python tools/convert_pcnetc_pretrain.py
    ```

3. Train (taking COCOA for example).

    ```
    sh experiments/COCOA/pcnet_c/train.sh
    ```

4. Monitoring status and visual results using tensorboard.


## Bibtex

```
@inproceedings{zhan2020self,
 author = {Zhan, Xiaohang and Pan, Xingang and Dai, Bo and Liu, Ziwei and Lin, Dahua and Loy, Chen Change},
 title = {Self-Supervised Scene De-occlusion},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 month = {June},
 year = {2020}
}
```
