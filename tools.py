import os
import random

def getDataset(path: str, shuffle_images=True) -> list:
    imgs_target_label = []
    data = sorted(os.listdir(path))

    for idx in range(0, len(data), 2):
        img = os.path.join(path, data[idx])
        target = os.path.join(path, data[idx + 1])
        label = data[idx].split("_")[0]
        imgs_target_label.append((img, target, label))

    if shuffle_images:
        random.shuffle(imgs_target_label)

    return imgs_target_label