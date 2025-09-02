import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from imgaug import augmenters as iaa

ROOT_DIR = r"C:\Users\Yi Su\UF\Mask\Mask"

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class ButterflyConfig(Config):
    NAME = "butterfly"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 2 #background + butterfly

    STEPS_PER_EPOCH = 5

    DETECTION_MIN_CONFIDENCE = 0.9

    

class ButterflyDataset(utils.Dataset):

    def load_butterflies(self, dataset_dir, subset):
        print("Loading Database")
        assert subset in ["train", "val"]
        coco = COCO("{}/{}/coco_annotation.json".format(dataset_dir, subset))

        image_dir = "{}/{}/images".format(dataset_dir, subset)

        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]["file_name"]),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None
                ))
            )
        print("Loaded Database")


    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        
        if image_info["source"] != "coco":
            return super(ButterflyDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id'])
            )
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])

                if m.max() < 1:
                    continue

                if annotation['iscrowd']:
                    class_id *= -1

                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)

                instance_masks.append(m)
                class_ids.append(class_id)
        
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super().load_mask(image_id)
        

    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']

        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        return rle
        

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(model, dataset_dir):
    dataset_train = ButterflyDataset()
    dataset_train.load_butterflies(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = ButterflyDataset()
    dataset_val.load_butterflies(dataset_dir, "val")
    dataset_val.prepare()

    augmentation = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05*255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ),
            iaa.AddToHueAndSaturation((-20, 20)),
        ],
        random_order=True
    )

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads',
                augmentation=augmentation)


if __name__ == "__main__":
    config = ButterflyConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])

    try:
        train(model, DATASET_DIR)
    except Exception as e:
        print("Error", e)
    print("Done")