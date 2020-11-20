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
        self.end = (int(np.log2(resolution_end)) - 2) * 2 + 2

        self.resolution_start_default = self.resolution_start
        self.resolution_end_default = self.resolution_end

        self.K = K
        self.N = N
        # self.d_avg, self.d_max, self.d_std = self.compute_w_distribution()
        self.w_avg = self.w_stds = None

        self.w_avgs = {}
        self.w_stds = {}

        for start in (4, 8, 16, 32, 64):
            end = start * 8
            self.set_resolution(start, start * 8)
            w_avg, w_std = self.compute_w_distribution(5000)
            self.w_avgs[start, end] = w_avg
            self.w_stds[start, end] = w_std

        self.w_avg = self.w_avgs[self.resolution_start, self.resolution_end]
        self.w_std = self.w_stds[self.resolution_start, self.resolution_end]

        self.set_resolution(self.resolution_start_default, self.resolution_end_default)
        print(self.start, self.end)

    def set_resolution(self, resolution_start, resolution_end):
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        self.start = (int(np.log2(resolution_start)) - 2) * 2
        self.end = (int(np.log2(resolution_end)) - 2) * 2 + 2

        if self.w_avg is not None:
            self.w_avg = self.w_avgs[resolution_start, resolution_end]
            self.w_std = self.w_stds[resolution_start, resolution_end]


    def compute_w_distribution(self, num_points=5000):
        mapping, synthesis = self.mapping, self.synthesis
        start, end = self.start, self.end

        z = np.random.random((num_points, 512))
        w = mapping.predict(z) # N, 18, 512
        return np.mean(w[start:end]), np.std(w[start:end])


    def find_anchors(self, N, num_std=2, batch_size=100):

        start, end = self.start, self.end
        w_avg, w_std = self.w_avg, self.w_std
        z = np.random.random((1, 512))
        mapping, synthesis = self.mapping, self.synthesis
        w_anchors = [mapping.predict(z)]
        z_anchors = [z]

        while len(w_anchors) < N:
            z_new = np.random.random((batch_size, 512))
            w_new = mapping.predict(z_new)
            dists = np.array([np.linalg.norm(w_new[:, start:end, :] - w[:, start:end, :], axis=(1, 2)) for w in w_anchors]).flatten()

            ids = dists > num_std * w_std

            if np.sum(ids) == 0:
                continue
            # randomly select a point that pass
            id = np.argsort(~ids)[0]

            w_anchors.append(w_new[id:id + 1])
            z_anchors.append(z_new[id:id + 1])

        # check anchor distance:
        dists = []
        for i in range(len(w_anchors)):
            for j in range(i+1, len(w_anchors)):
                w_i = w_anchors[i][:, start:end, :]
                w_j = w_anchors[j][:, start:end, :]
                d = np.linalg.norm(w_i - w_j)
                if d < num_std * w_std:
                    print("fail anchors")
                dists.append(d)
        # print(np.mean(dists), w_std, "distance distribution")
        return w_anchors, z_anchors

    def sample_around_anchors(self, K, N, w_anchors, z_anchors, num_std=0.3, noise_std=0.00015, batch_size=8000):

        mapping, synthesis = self.mapping, self.synthesis
        start, end = self.start, self.end
        w_avg, w_std = self.w_avg, self.w_std

        eye = np.eye(N)
        images = []
        labels = []
        for n, (w_anchor, z_anchor) in enumerate(zip(w_anchors, z_anchors)):
            w_nears = w_anchor
            num_run = 0
            while w_nears.shape[0] < K:
                num_run += 1
                # temporary set noise std to be 0.01; can optimize to increase sampling efficient
                z_noise = z_anchor + np.random.normal(0, 1, (batch_size, z_anchor.shape[-1])) * noise_std
                w_noise = mapping.predict(z_noise)
                dists = np.linalg.norm(w_noise[:, start:end, :] - w_anchor[:, start:end, :], axis=(1, 2))
                ids = dists < num_std * w_std
                w_nears = np.concatenate([w_nears, w_noise[ids]], axis=0)

                if num_run > 2:
                    print("warning: standard deviation set to be too high")
                    print("minimum distance: {}, noise level: {}" .format(np.min(dists), num_std * w_std))
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

    def shuffle_resolution(self, eps=1):
        rnd = np.random.random()
        if rnd > eps:
            self.set_resolution(self.resolution_start_default, self.resolution_end_default)
            return


        start = np.random.choice((4, 8, 16, 32, 64))
        weights = np.array([1, 1, 1, 1, 1])
        weights = weights / np.sum(weights)

        end = start * 8
        self.set_resolution(start, end)



    def sample_batch(self, batch_size, K, N, num_std=0.1, noise_std=0.005, shuffle=True, swap=False, h=64, w=64, shuffle_resolutoin=False):
        if shuffle_resolutoin:
            self.shuffle_resolution()
        # images, labels = self.sample_around_anchors(K, N)
        w_anchors, z_anchors = self.find_anchors(N)
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