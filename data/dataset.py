import os
import pdb

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms


def get_mot_loader(dataset, test, data_dir="data", workers=4, size=(800, 1440)):
    # Different dataset paths
    if dataset == "mot17":
        direc = "mot"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    elif dataset == "mot20":
        direc = "MOT20"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    elif dataset == "dance":
        direc = "dancetrack"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            annotation = "val.json"
            name = "val"
    elif dataset in ["MOT17-01-FRCNN", "MOT17-03-FRCNN", "MOT17-06-FRCNN",
                     "MOT17-07-FRCNN", "MOT17-08-FRCNN", "MOT17-12-FRCNN", "MOT17-14-FRCNN"]:
        direc = "mot"
        name = "test"
        annotation = f"anno_by_seq/test-{dataset}.json"
    elif dataset in ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                     "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]:
        direc = "mot"
        name = "train"
        annotation = f"anno_by_seq/val_half-{dataset}.json"
    else:
        raise RuntimeError("Specify path here.")

    # Same validation loader for all MOT style datasets
    valdataset = MOTDataset(
        data_dir=os.path.join(data_dir, direc),
        json_file=annotation,
        img_size=size,
        name=name,
        preproc=ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        ## preproc=ValTransform(rgb_means=(0.0, 0.0, 0.0), std=(1.0, 1, 1.0),)
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)
    dataloader_kwargs = {"num_workers": workers, "pin_memory": True, "sampler": sampler, "batch_size": 1}
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return val_loader


class MOTDataset(torch.utils.data.Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        self.input_dim = img_size
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)

        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img :
                img_info = (height, width, frame_id, video_id, file_name)
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)
        tensor, target = self.preproc(img, target, self.input_dim)
        return (tensor, img), target, img_info, img_id


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r
