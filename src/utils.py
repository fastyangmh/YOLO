#import
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join


#def
def visualize_data(image, bboxes, classes):
    channels, height, width = image.shape
    labels = bboxes[:, -5].astype(int)
    bboxes = bboxes[:, -4:]
    #convert bboxes(xywh) to xyxy
    xyxy = torch.zeros_like(bboxes) if isinstance(
        bboxes, torch.Tensor) else np.zeros_like(bboxes)
    xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
    xyxy[:, [0, 2]] *= width
    xyxy[:, [1, 3]] *= height
    xyxy = xyxy.astype(int)

    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)
    for l, pt in zip(labels, xyxy):
        cv2.rectangle(img=image,
                      pt1=pt[:2],
                      pt2=pt[2:],
                      color=(0, 0, 255),
                      thickness=2)
        cv2.putText(img=image,
                    text=classes[l],
                    org=pt[:2] - 10,
                    fontFace=0,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2)
    image = image[..., ::-1]
    plt.imshow(image)
    plt.show()


def avg_iou(bboxes, clusters):
    return np.mean(
        [np.max(iou(bboxes[i], clusters)) for i in range(bboxes.shape[0])])


def iou(box, clusters):
    x = np.minimum(box[0], clusters[:, 0])
    y = np.minimum(box[1], clusters[:, 1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    return intersection / (box_area + cluster_area - intersection)


def kmeans(bboxes, n_clusters, dist=np.median):
    num = bboxes.shape[0]
    assert num > n_clusters, f'please increase the samples of the training dataset or decrease n_clusters.\nthe numbers of training dataset samples: {num}\nthe n_clusters: {n_clusters}'
    distances = np.empty(shape=(num, n_clusters))
    last_clusters = np.zeros(num)
    clusters = bboxes[np.random.choice(num, size=n_clusters, replace=False)]
    while True:
        for idx in range(num):
            distances[idx] = iou(bboxes[idx], clusters)
        nearest_clusters = np.argmax(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster_idx in range(n_clusters):
            points = bboxes[nearest_clusters == cluster_idx]
            if len(points):
                clusters[cluster_idx] = dist(points, axis=0)
        last_clusters = nearest_clusters
    return clusters


def get_anchors(predefined_dataset,
                root,
                dataset_class,
                n_clusters,
                image_size,
                n_iter=3):
    if predefined_dataset:
        root = join(root, predefined_dataset)
        train_dataset = dataset_class(root=root,
                                      train=True,
                                      transform=None,
                                      target_transform=None,
                                      download=True,
                                      image_size=image_size)
    else:
        root = join(root, 'train')
        train_dataset = dataset_class(root=root,
                                      transform=None,
                                      target_transform=None)

    bboxes = []
    for _, y in train_dataset:
        bboxes.append(y[:, -2:])  #get the width and height of box
    bboxes = np.vstack(bboxes)
    accuracy = 0
    for _ in range(n_iter):
        clusters = kmeans(bboxes=bboxes, n_clusters=n_clusters)
        acc = avg_iou(bboxes=bboxes, clusters=clusters)
        if acc > accuracy:
            accuracy = acc
            anchor_box = clusters[np.argsort(clusters[:, 0])]
            anchors = np.round((anchor_box * image_size)).astype(int)
    return anchors