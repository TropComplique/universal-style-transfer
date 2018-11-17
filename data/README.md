# Data preparation

### COCO

1. Download train and val images [here](http://cocodataset.org/#download)
2. Get [COCO API](https://github.com/cocodataset/cocoapi)

3. Convert tfrecords  
  ```
  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/coco_train/images/ \
      --annotations_dir=/mnt/datasets/dan/coco_train/annotations/ \
      --output=/mnt/datasets/dan/coco_train_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1000

  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/coco_val/images/ \
      --annotations_dir=/mnt/datasets/dan/coco_val/annotations/ \
      --output=/mnt/datasets/dan/coco_val_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1
  ```  
