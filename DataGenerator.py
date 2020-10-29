from skimage.transform import resize
import random
import numpy as np


class DataGenerator:
    def __init__(self, mapping, synthesis, K=2, N=5, resolution_start=16, resolution_end=256):
        self.mapping = mapping
        self.synthesis = synthesis
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        self.K = K
        self.N = N
        self.d_avg, self.d_max, self.d_std = self.compute_w_distribution()

    def compute_w_distribution(self, num_points=1000):
        mapping, synthesis = self.mapping, self.synthesis
        resolution_start, resolution_end = self.resolution_start, self.resolution_end

        start = int(np.log2(resolution_start)) - 2
        end = int(np.log2(resolution_end)) - 2
        z = np.random.random((num_points, 512))
        weights = mapping.predict(z)[:, start:end, :].reshape(num_points, -1)
        dists = []
        for i in range(num_points):
            w_curr = weights[i:i + 1]
            d = np.linalg.norm(weights - w_curr, axis=1)
            dists.append(d)
        dists = np.array(dists).flatten()
        d_avg = np.mean(dists)
        d_max = np.max(dists)
        d_min = np.min(dists)
        d_std = np.std(dists)
        print("average distance: {}, max distance: {}, std: {}".format(d_avg, d_max, d_std))
        return d_avg, d_max, d_std

    def compute_distance(self, w_batch, w):
        resolution_start, resolution_end = self.resolution_start, self.resolution_end
        start = int(np.log2(resolution_start)) - 2
        end = int(np.log2(resolution_end)) - 2
        w_batch = w_batch[:, start:end, :].reshape(w_batch.shape[0], -1)
        w = w[:, start:end, :].reshape(1, -1)
        d = np.linalg.norm(w_batch - w, axis=1)
        return d

    def rejection_sample(self, num_std=2, num_points_per_sample=100):
        N = self.N
        d_avg, d_max, d_std = self.d_avg, self.d_max, self.d_std
        z = np.random.random((1, 512))
        mapping, synthesis = self.mapping, self.synthesis
        weights = [mapping.predict(z)]
        zs = [z]
        num_points_per_sample = 100
        i = 0
        while len(weights) < N:
            z_new = np.random.random((num_points_per_sample, 512))
            w_new = mapping.predict(z_new)
            dists = np.min(np.array([self.compute_distance(w_new, w) for w in weights]), axis=0)
            if np.max(dists) < d_avg + d_std * num_std:
                continue
            idx_sort = np.argsort(-dists)

            for idx in idx_sort:
                if dists[idx] > d_avg + d_std * num_std:
                    weights.append(w_new[idx:idx + 1])
                    zs.append(z_new[idx:idx + 1])
                    if len(weights) == N:
                        break
        # check to verify
        # for i in range(N):
        #    d = np.min([self.compute_distance(weights[i], w) for w in weights[:i] + weights[i+1:]], axis=0)
        #    print(d)
        return weights, zs

    def sample_around_anchors(self):
        resolution_start, resolution_end = self.resolution_start, self.resolution_end
        mapping, synthesis = self.mapping, self.synthesis
        w_anchors, zs = self.rejection_sample()
        start = int(np.log2(resolution_start)) - 2
        end = int(np.log2(resolution_end)) - 2

        N = len(w_anchors)
        K = self.K

        images = []
        labels = []
        eye = np.eye(N)
        for n, w_anchor in enumerate(w_anchors):
            z = np.random.random((K, 512))
            w_random = mapping.predict(z)
            w_anchor = np.repeat(w_anchor, K, axis=0)
            w_mix = np.concatenate([w_random[:, :start], w_anchor[:, start:end], w_random[:, end:]], axis=1)
            images.append(synthesis.predict(w_mix))
            labels.append(np.repeat(eye[n, :][None, :], K, axis=0))

        return images, labels

    def resize(self, images, h_new, w_new):
        N = len(images)
        images_out = []
        for imgs in images:
            imgs_out = []
            for i in range(imgs.shape[0]):
                img_i = np.rollaxis(imgs[i], 0, 3)
                imgs_out.append(resize(img_i, (h_new, w_new)))
            images_out.append(np.array(imgs_out))
        return np.array(images_out)

    def sample_batch(self, batch_size, shuffle=True, swap=False, h=64, w=64):
        images, labels = self.sample_around_anchors()
        images = self.resize(images, h, w).astype(np.float32)
        labels = np.array(labels)

        image_batches = []
        label_batches = []

        N, K = self.N, self.K

        eye = np.eye(N)
        for _ in range(batch_size):

            if shuffle:
                images = np.reshape(images, (N, K, -1))
                batch = np.concatenate([labels, images], 2)

                for p in range(K):
                    np.random.shuffle(batch[:, p])

                labels = batch[:, :, :N]
                images = batch[:, :, N:]

            if swap:
                # K, N, -1
                images = np.swapaxes(images, 0, 1)
                # K, N, N
                labels = np.swapaxes(labels, 0, 1)
            image_batches.append(np.reshape(images, (1, N, K, h, w, 3)))
            label_batches.append(labels)

        all_image_batches = np.stack(image_batches, axis=0)
        all_label_batches = np.stack(label_batches, axis=0)

        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)
