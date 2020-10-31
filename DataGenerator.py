from skimage.transform import resize
import random
import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, mapping, synthesis, K=2, N=5, resolution_start=32, resolution_end=128):
        self.mapping = mapping
        self.synthesis = synthesis
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        self.start = (int(np.log2(resolution_start)) - 2) * 2
        self.end = (1 + int(np.log2(resolution_start)) - 2) * 2
        self.K = K
        self.N = N
        # self.d_avg, self.d_max, self.d_std = self.compute_w_distribution()
        self.w_avg, self.w_std = self.compute_w_distribution(5000)

    def set_resolution(self, resolution_start, resolution_end):
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        self.start = (int(np.log2(resolution_start)) - 2) * 2
        self.end = (1 + int(np.log2(resolution_start)) - 2) * 2

    def compute_w_distribution(self, num_points=5000):
        mapping, synthesis = self.mapping, self.synthesis
        z = np.random.random((num_points, 512))
        w = mapping.predict(z)
        std = np.std(w, axis=0)
        mu = np.mean(w, axis=0)
        return mu, std

    def find_anchors(self, K, N, num_std=2, batch_size=100):

        start, end = self.start, self.end
        w_avg, w_std = self.w_avg, self.w_std
        z = np.random.random((1, 512))
        mapping, synthesis = self.mapping, self.synthesis
        w_anchors = [mapping.predict(z)]
        z_anchors = [z]

        while len(w_anchors) < N:
            z_new = np.random.random((batch_size, 512))
            w_new = mapping.predict(z_new)
            dists = np.array([np.linalg.norm(w_new - w, axis=-1) for w in w_anchors])

            ids = np.all(np.all(dists[:, :, start:end] > num_std * w_std[start:end], axis=-1), axis=0)

            # randomly select a point that pass
            id = np.argsort(ids)[-1]

            w_anchors.append(w_new[id:id + 1])
            z_anchors.append(z_new[id:id + 1])

        return w_anchors, z_anchors

    def sample_around_anchors(self, K, N, w_anchors, z_anchors, num_std=0.1, noise_std=0.005, batch_size=500):

        mapping, synthesis = self.mapping, self.synthesis
        start, end = self.start, self.end
        w_avg, w_std = self.w_avg, self.w_std

        eye = np.eye(N)
        images = []
        labels = []
        for n, (w_anchor, z_anchor) in enumerate(zip(w_anchors, z_anchors)):
            # print(w_anchor.shape, z_anchor.shape, "w_anchor, z_anchor")
            w_nears = w_anchor
            num_run = 0
            while w_nears.shape[0] < K:
                num_run += 1
                # temporary set noise std to be 0.01; can optimize to increase sampling efficient
                z_noise = z_anchor + np.random.normal(0, noise_std, (batch_size, z_anchor.shape[-1]))
                w_noise = mapping.predict(z_noise)
                dists = np.linalg.norm(w_noise - w_anchors, axis=-1)
                ids = np.all(dists[:, start:end] < num_std * w_std[start:end], axis=-1)

                w_nears = np.concatenate([w_nears, w_noise[ids]], axis=0)
                if num_run > 2:
                    print("warning: standard deviation set to be too high")
                if num_run == 1 and w_nears.shape[0] == batch_size:
                    print("warning: noise standard deviation set to be too low")
            w_nears = w_nears[:K]

            z = np.random.random((K, 512))
            w_random = mapping.predict(z)

            w_mix = np.concatenate([w_random[:, :start], w_nears[:, start:end], w_random[:, end:]], axis=1)

            images.append(synthesis.predict(w_mix))
            labels.append(np.repeat(eye[n, :][None, :], K, axis=0))
        return images, labels

    def resize(self, images, h_new, w_new):
        images_out = []
        for img_k in images:
            img_k = np.rollaxis(img_k, 1, 4)

            img_k_resize = tf.image.resize(img_k, [h_new, w_new])

            images_out.append(img_k_resize.numpy())

        # tensorflow bug, very slow if don't convert tensor to numpy first
        images_out = np.array(images_out)

        return images_out

    def sample_batch(self, batch_size, K, N, num_std=0.1, noise_std=0.005, shuffle=True, swap=False, h=64, w=64):
        # images, labels = self.sample_around_anchors(K, N)
        w_anchors, z_anchors = self.find_anchors(K, N)
        images, labels = self.sample_around_anchors(K, N, w_anchors, z_anchors, num_std, noise_std)

        images = self.resize(images, h, w)
        labels = np.array(labels)
        image_batches = []
        label_batches = []
        eye = np.eye(N)
        for _ in range(batch_size):

            if shuffle:
                images = np.reshape(images, (N, K, -1))
                batch = np.concatenate([labels, images], 2)

                for p in range(K):
                    np.random.shuffle(batch[:, p])

                labels = batch[:, :, :N]
                images = batch[:, :, N:]
                images = np.reshape(images, (N, K, h, w, 3))

            if swap:
                # K, N, -1
                images = np.swapaxes(images, 0, 1)
                # K, N, N
                labels = np.swapaxes(labels, 0, 1)
            image_batches.append(images)
            label_batches.append(labels)
        all_image_batches = np.stack(image_batches, axis=0)
        all_label_batches = np.stack(label_batches, axis=0)

        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)
