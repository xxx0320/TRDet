# TRDet

## **Abstract**    

This project is based on [R<sup>2</sup>CNN](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow), and completed by [BaochaiPeng](https://github.com/xxx0320) .

## Requirements

1、tensorflow >= 1.2     
2、cuda8.0     
3、python2.7 (anaconda2 recommend)    
4、[opencv(cv2)](https://pypi.org/project/opencv-python/)     
5、[tfplot](https://github.com/wookayin/tensorflow-plot)     

## Download Model

1、please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to data/pretrained_weights.     
2、please download [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) pre-trained model on Imagenet, put it to data/pretrained_weights/mobilenet.      
3、please download [trained model](https://github.com/DetectionTeamUCAS/Models/tree/master/R2CNN-Plus-Plus_Tensorflow) by this project, put it to output/trained_weights.

## Data Prepare

1、crop data\, reference:

```  
cd $PATH_ROOT/data/io/data
python train_crop.py 
python val_crop.py
```

2、data format

```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│    ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```

## Compile

```  
cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```



## Train

1、If you want to train your own data, please note:  

```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to $PATH_ROOT/data/io/read_tfrecord.py 
```

2、make tfrecord

```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/VOCdevkit/VOCdevkit_train/' 
                                   --xml_dir='Annotation'
                                   --image_dir='JPEGImages'
                                   --save_name='train' 
                                   --img_format='.png' 
                                   --dataset='dataset'
```

3、train

```  
cd $PATH_ROOT/tools
python train.py
```

## Eval

```  
python eval.py --img_dir='/PATH/TO/dataset/IMAGES/' 
               --image_ext='.png' 
               --test_annotation_path='/PATH/TO/TEST/ANNOTATION/'
               --gpu='0'
```

## Tensorboard

```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
```

