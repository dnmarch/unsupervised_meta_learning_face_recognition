"""Data Generator"""
import os
import numpy as np
import pandas as pd
import random
from scipy import misc
import imageio
import cv2

######################## Directory ########################
WORK_DIR = "/content"
celebA_dir = os.path.join(WORK_DIR, 'celebA_anno')
celebA_HQ_dir = os.path.join(WORK_DIR, 'CelebAMask-HQ')
img_align_dir = os.path.join(WORK_DIR, 'img_align_celeba')


###########################################################

def read_img(img_path, resize='cut'):
    """cv2 reads BGR, Pic is RGB, reorder the the 2 axies"""
    img = imageio.imread(img_path)
    assert img is not None
    img = img_resize(img, resize)
    img = img.astype(np.float32) / 255
    img = 1 - img
    return img


def img_resize(img, how='cut'):
    """The CelebA img is (218, 178, 3), we can either resize to (256, 256)
    or cut 0 dim to [178, 178, 3]"""
    how = how.lower()
    assert how in ['cut', 'resize']
    if how == 'cut':
        img = img[20:198, :, :]
        assert img.shape == (178, 178, 3)
    if how == 'resize':
        img = cv2.resize(img, (256, 256))
    return img


def get_labels_df():
    id_celebA = pd.read_table(os.path.join(celebA_dir, "identity_CelebA.txt"),
                              delim_whitespace=True,
                              names=['file', 'idx']
                              )
    mapping = pd.read_table(os.path.join(celebA_HQ_dir, "CelebA-HQ-to-CelebA-mapping.txt"),
                            delim_whitespace=True,
                            names=['hq_idx', 'idx', 'file'], skiprows=1
                            )

    # Get a list of HQ images that also exist in CelebA by checking the idx
    # Left join and see file is not Nan
    mapping = mapping.merge(id_celebA, how='left', on=['idx'], suffixes=('_hd', ''))
    mapping = mapping[~mapping.file.isna()]

    imgs = mapping.groupby('idx').apply(lambda x: x.shape[0]). \
        reset_index().rename(columns={0: 'counter'})

    imgs = imgs.merge(mapping, how='left', on=['idx'])
    # Find all idx with more than 10 images set to dataset ~ to omniglot
    multi_face = imgs[imgs.counter >= 10].copy()
    multi_face.sort_values('idx', inplace=True)

    assert len(multi_face.file.unique()) == len(multi_face)

    # Create Labels by re-map idx
    label_dict = {}
    for i, idx in enumerate(multi_face.idx.unique()):
        label_dict[idx] = i

    multi_face['label'] = [label_dict[k] for k in multi_face.idx.values]
    # aligned images are in png format
    # multi_face.file = [x.replace('jpg', 'png') for x in multi_face.file]
    return multi_face


def get_images(df, hq_idx, labels, n_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    """
    if n_samples is not None:
        sampler = lambda x: random.sample(x, n_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(img_align_dir, image))
                     for i, hd_i in zip(labels, hq_idx)
                     for image in sampler(list(df[df.hq_idx == hd_i]['file'].values))
                     ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataLoader(object):
    """
    Data Generator capable of generating batches of CelebA data.
    Generated image in [B, N, K, h, w, c]
    """

    def __init__(self, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class, df):
        """
        Args:
          num_classes: Number of classes for classification (N-way)
          num_samples_per_class: num samples to generate per class in one batch
          num_meta_test_classes: Number of classes for classification (N-way) at meta-test time
          num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
          batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        self.df = get_labels_df()  # main df contains [idx, counter, hq_idx, file_hd, file, label]

        WORK_DIR = "/Users/zhejianpeng/Google Drive/GaTech/AI_Cert/CS330_Multitask_MetaLearning/Final_Project"
        celebA_dir = os.path.join(WORK_DIR, 'celebA_anno')
        celebA_HQ_dir = os.path.join(WORK_DIR, 'CelebAMask-HQ')
        img_align_dir = os.path.join(WORK_DIR, 'img_align_celeba')

        self.img_size = (178, 178, 3)  # or (256, 256, 3)

        self.resize_method = 'cut'  # self.img_size
        self.dim_output = self.num_classes

        # Construct Meta-Train, Meta-Test, Meta-Val
        face_idx = list(self.df.hq_idx.unique())  # Each hd_id map to multiple images of that face
        random.seed(123)
        np.random.seed(123)
        random.shuffle(face_idx)
        num_val = 100
        num_train = 850
        self.metatrain_face_idx = face_idx[:num_train]
        self.metaval_face_idx = face_idx[num_train: num_train + num_val]
        self.metatest_face_idx = face_idx[num_train + num_val:]
        print(f"{num_train} Train, {num_val} Val {len(self.metatest_face_idx)} Test")

    def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
          shuffle: randomly shuffle classes or not
          swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
          A a tuple of (1) Image batch and (2) Label batch where
          image batch has shape [B, N, K, h*w*c] and label batch has shape [B, N, K, N] if swap is False
          where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "meta_train":
            face_idx = self.metatrain_face_idx
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            face_idx = self.metaval_face_idx
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            face_idx = self.metatest_face_idx
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        all_image_batches, all_label_batches = [], []
        for i in range(batch_size):
            sampled_character_folders = random.sample(
                face_idx, num_classes)
            labels_and_images = get_images(self.df, face_idx, range(
                num_classes), n_samples=num_samples_per_class, shuffle=False)
            labels = [li[0] for li in labels_and_images]
            images = [read_img(
                li[1], self.resize_method) for li in labels_and_images]
            images = np.stack(images)
            labels = np.array(labels).astype(np.int32)
            labels = np.reshape(
                labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes, dtype=np.float32)[labels]
            images = np.reshape(
                images, [num_classes, num_samples_per_class, -1])

            batch = np.concatenate([labels, images], 2)
            if shuffle:
                for p in range(num_samples_per_class):
                    np.random.shuffle(batch[:, p])

            labels = batch[:, :, :num_classes]
            images = batch[:, :, num_classes:]

            if swap:
                labels = np.swapaxes(labels, 0, 1)
                images = np.swapaxes(images, 0, 1)

            # Since we flattend the images previously, we want to shape it back
            # images = np.reshape(
            #     images, [num_classes,
            #              num_samples_per_class,
            #              self.img_size[0],
            #              self.img_size[1],
            #              self.img_size[2]]
            #             )

            all_image_batches.append(images)
            all_label_batches.append(labels)
        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)
        return all_image_batches, all_label_batches


class TestDataLoader(DataLoader):
    """Inheriated from DataLoader, Set all images to testing images"""

    def __init__(self, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class, df):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        self.df = get_labels_df()  # main df contains [idx, counter, hq_idx, file_hd, file, label]

        WORK_DIR = "/Users/zhejianpeng/Google Drive/GaTech/AI_Cert/CS330_Multitask_MetaLearning/Final_Project"
        celebA_dir = os.path.join(WORK_DIR, 'celebA_anno')
        celebA_HQ_dir = os.path.join(WORK_DIR, 'CelebAMask-HQ')
        img_align_dir = os.path.join(WORK_DIR, 'img_align_celeba')

        self.img_size = (178, 178, 3)  # or (256, 256, 3)

        self.resize_method = 'cut'  # self.img_size
        self.dim_output = self.num_classes

        face_idx = list(self.df.hq_idx.unique())  # Each hd_id map to multiple images of that face
        self.metatest_face_idx = face_idx
        print(f"Test on {len(self.metatest_face_idx)} classes | Total {len(self.df)} images")