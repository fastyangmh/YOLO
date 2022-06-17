# import
from functools import partial
import numpy as np
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyVOCDetection, MyImageFolder, YOLOImageLightningDataModule
from typing import Optional, Callable
from src.utils import get_anchors
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from glob import glob
from os.path import join
from typing import Any
from PIL import Image
import random


# def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_class = MyImageFolder
    dataset_class = partial(dataset_class,
                            image_size=project_parameters.image_size)
    if project_parameters.anchors is None:
        #calculate anchors
        project_parameters.anchors = get_anchors(
            predefined_dataset=project_parameters.predefined_dataset,
            root=project_parameters.root,
            dataset_class=dataset_class,
            n_clusters=project_parameters.n_clusters,
            image_size=project_parameters.image_size)
    else:
        #format anchors
        anchors = project_parameters.anchors
        if isinstance(anchors, str):
            anchors = [int(v) for v in anchors.split(',')]
        assert len(
            project_parameters.anchors
        ) == project_parameters.n_clusters * 2, 'please check the n_clusters and numbers of anchor argument.\nanchors: {}\nvalid: {}'.format(
            project_parameters.anchors, project_parameters.n_clusters * 2)
        anchors = np.array(
            [anchors[idx:idx + 2] for idx in range(0, len(anchors), 2)])
        project_parameters.anchors = anchors
    return YOLOImageLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        dataset_class=dataset_class)


#class
class MyImageFolder(VisionDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader=Image.open,
                 image_size=None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        images = []
        for ext in IMG_EXTENSIONS:
            images += glob(join(root, f'*{ext}'))
        self.images = images
        self.targets = [v[:-3] + 'txt' for v in self.images]
        self.classes = np.loadtxt(join(root, '../classes.txt'),
                                  dtype=str,
                                  delimiter='\n',
                                  ndmin=1).tolist()
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        self.loader = loader
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        img = self.loader(self.images[index])
        img = np.array(img)  #the img dimension is (height, width, channels)
        bboxes = np.loadtxt(
            self.targets[index], delimiter=' ', ndmin=2
        )  #the bboxes dimension is (n_objects, bounding_box_info) [[c, x, y, w, h]]
        if self.transform:
            if len(bboxes) == 0:
                img, _ = self.transform(image=img,
                                        bboxes=[[1e-10] * 4 + [0]]).values()
            else:
                bboxes = bboxes[:,
                                [1, 2, 3, 4,
                                 0]]  #[[c, x, y, w, h]] -> [[x, y, w, h, c]]
                img, bboxes = self.transform(image=img, bboxes=bboxes).values()
                bboxes = np.array(bboxes)
                bboxes = bboxes[..., None] if len(bboxes) == 0 else bboxes
                #[[x, y, w, h, c]] -> [[c, x, y, w, h]]
                bboxes = bboxes[:, [4, 0, 1, 2, 3]]
            size = max(img.shape[:-1])
            assert size == self.image_size, f'the transformed image size not equal image_size in config.yml.\ntransformed image size: {size}\nimage_size in config.yml: {self.image_size}'
        if self.target_transform and len(bboxes):
            labels = bboxes[:, 0].astype(int)
            labels = self.target_transform(labels)
            bboxes = np.append(arr=labels, values=bboxes[:, 1:], axis=-1)
        if len(bboxes):
            #add target image index for build_targets()
            bboxes = np.append(arr=np.zeros(shape=(len(bboxes), 1)),
                               values=bboxes,
                               axis=-1)
        else:
            #image_index + one hot encoder length + xywh if self.target_transform
            #else image_index + c + xywh
            l = 1 + len(
                self.classes) + 4 if self.target_transform else 1 + 1 + 4
            bboxes = np.zeros(shape=(0, l))
        img = img.transpose(
            2, 0, 1)  #transpose dimension to (channels, width, height)
        bboxes = bboxes.astype(np.float32)
        return img, bboxes

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.images)),
                                  k=max_samples)
            self.images = np.array(self.images[index])
            self.targets = np.array(self.targets[index])


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print('the dimension of target: {}'.format(y.shape))
