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
        self.K = K
        self.N = N
        self.d_avg, self.d_max, self.d_std = self.compute_w_distribution()

    def set_resolution(self, resolution_start, resolution_end):
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        
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

    def rejection_sample(self, K, N, num_std=2, num_points_per_sample=100):
        
        d_avg, d_max, d_std = self.d_avg, self.d_max, self.d_std
        z = np.random.random((1, 512))
        mapping, synthesis = self.mapping, self.synthesis
        weights = [mapping.predict(z)]
        z_anchors = [z]
        
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
                    z_anchors.append(z_new[idx:idx + 1])
                    # print("find a point")
                    if len(weights) == N:
                        break
        
        # check to verify
        # for i in range(N):
        #    d = np.min([self.compute_distance(weights[i], w) for w in weights[:i] + weights[i+1:]], axis=0)
        #    print(d)
        return weights, z_anchors

    def sample_around_anchors2(self, K, N):
        resolution_start, resolution_end = self.resolution_start, self.resolution_end
        mapping, synthesis = self.mapping, self.synthesis
        w_anchors, zs = self.rejection_sample(K, N)
        start = int(np.log2(resolution_start)) - 2
        end = int(np.log2(resolution_end)) - 2

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
    

        
    def find_anchors(self, K, N, num_std=2, num_points_per_sample=100):
        
        d_avg, d_max, d_std = self.d_avg, self.d_max, self.d_std
        z = np.random.random((1, 512))
        mapping, synthesis = self.mapping, self.synthesis
        weights = [mapping.predict(z)]
        z_anchors = [z]
        
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
                    z_anchors.append(z_new[idx:idx + 1])
                    # print("find a point")
                    if len(weights) == N:
                        break
        
        # check to verify
        # for i in range(N):
        #    d = np.min([self.compute_distance(weights[i], w) for w in weights[:i] + weights[i+1:]], axis=0)
        #    print(d)
        return weights, z_anchors

    def sample_around_anchors(self, K, N, w_anchors, z_anchors, num_std = 0.1, noise_std = 0.005, batch_size = 500):
        resolution_start, resolution_end = self.resolution_start, self.resolution_end
        mapping, synthesis = self.mapping, self.synthesis
        d_avg, d_max, d_std = self.d_avg, self.d_max, self.d_std
        start = int(np.log2(resolution_start)) - 2
        end = int(np.log2(resolution_end)) - 2

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
                z_noise = z_anchor + np.random.normal(0, 0.005, (batch_size, z_anchor.shape[-1]))
                w_noise = mapping.predict(z_noise)
                dists = self.compute_distance(w_noise, w_anchor)
                w_nears = np.concatenate([w_nears, w_noise[dists < num_std * d_std]], axis=0)
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
        #images, labels = self.sample_around_anchors(K, N)
        w_anchors, z_anchors = self.find_anchors(K, N)
        images, labels = self.sample_around_anchors(K, N, num_std, noise_std, w_anchors, z_anchors)
        
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
