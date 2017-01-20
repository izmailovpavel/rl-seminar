import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label, convex_hull_image
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt



class ObjectDetectionFeatures:
    def __init__(self, env):
        self.env = env
        self.act = env.action_space
        self.find_bg()
        self.all_classes = self.find_all_obj_classes()

    def find_bg(self, proba=True, n_starts=20, max_samples_im_count=30000):
        images = []
        for i in range(n_starts):
            self.env.reset()
            while not self.env.ale.game_over():
                im = self.env.ale.getScreenGrayscale()
                im = im[:, :, 0]
                im = im.astype('uint8')
                images.append(im)
                self.env.step(self.act.sample())
            if len(images) > max_samples_im_count:
                break

        N = len(images)
        images[-1] = 255 * np.ones(images[0].shape, dtype='uint8')
        images = np.array(images)

        hst = np.apply_along_axis(func1d=np.bincount, axis=0, arr=images)
        if proba:
            bg_proba = hst / float(N)
            self.bg_proba = bg_proba
            self.proba_computed=True
            return bg_proba

        bg = np.argmax(hst, axis=0)
        return bg

    def binarization(self, image):
        if self.proba_computed:
            bg = self.bg_proba.copy()
        else:
            bg = self.find_bg()

        th = 0.05
        bg = np.reshape(bg, (256, -1))
        im = np.ravel(image)
        bin_im = bg[im, range(im.shape[0])] < th
        return bin_im.reshape((image.shape[0], image.shape[1]))

    def delete_small_obj(self, li, th=3):
        for i in np.unique(li):
            w = li == i
            if np.sum(w) < 3:
                li[w] = 0
        return li

    def get_object_labels(self, image):
        bin_im = self.binarization(image)

        labeled_im = label(bin_im)
        labeled_im = self.delete_small_obj(labeled_im)

        return labeled_im

    def find_all_obj_classes(self, n_starts=10, max_samples_im_count=5000):
        n_samples = 0
        last_changes = n_samples
        classes = set([])

        for i in range(n_starts):
            self.env.reset()
            while not self.env.ale.game_over():
                n_cl = len(classes)
                #if n_samples % 1000 == 0:
                #    print n_samples
                n_samples += 1
                image = self.env.ale.getScreenGrayscale()
                li = self.get_object_labels(image)
                for l in np.unique(li):
                    #waring
                    if l == 0:
                        continue

                    image = np.ravel(image)
                    li = np.ravel(li)
                    ind = np.where(li == l)
                    #print image[ind][0]
                    classes.add(image[ind][0])
                if len(classes) != n_cl:
                    last_changes = n_samples

                if n_samples - last_changes > 2000:
                    break

            if n_samples > max_samples_im_count:
                break

            if n_samples - last_changes > 2000:
                break

        return classes

    def compute_all_cm(self, im):
        cms = []
        li = label(im)
        for i in np.unique(li):
            if i == 0:
                continue
            w = li == i
            cms.append(list(center_of_mass(w)))
        return np.array(cms)

    def find_closest(self, image, c1, c2):
        w1 = image == c1
        w2 = image == c2

        if np.sum(w1 != 0) == 0:
            return -1, -image.shape[0] * 1.5, 0

        if np.sum(w2 != 0) == 0:
            return -1, image.shape[0] * 1.5, 0

        im1 = image.copy()
        im1[~w1] = 0
        cm1 = self.compute_all_cm(im1)

        im2 = image.copy()
        im2[~w2] = 0
        cm2 = self.compute_all_cm(im2)

        y_dif = np.repeat(cm1[:, 0][:, None], cm2.shape[0], axis=1) - np.repeat(cm2[:, 0][:, None], cm1.shape[0], axis=1).T
        x_dif = np.repeat(cm1[:, 1][:, None], cm2.shape[0], axis=1) - np.repeat(cm2[:, 1][:, None], cm1.shape[0], axis=1).T

        #print x_dif
        #print y_dif
        dist = np.sqrt(x_dif ** 2 + y_dif ** 2)
        ind = np.unravel_index(dist.argmin(), dist.shape)

        return ind, x_dif[ind[0], ind[1]], y_dif[ind[0], ind[1]]

    def get_distance_features(self, image):
        cl = self.all_classes
        cl = list(cl)
        new_features = []
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                _, x_dif, y_dif = self.find_closest(image, cl[i], cl[j])
                new_features.append(x_dif)
                new_features.append(y_dif)
        return np.array(new_features)

    def find_extr_points(self, bi):
        x_top = y_right = 0
        x_low = bi.shape[0]
        y_left = bi.shape[1]

        for i in range(bi.shape[0]):
            if np.sum(bi[i, :]):
                x_low = i
                break

        for i in range(bi.shape[0] - 1, -1, -1):
            if np.sum(bi[i, :]):
                x_top = i
                break

        for i in range(bi.shape[1]):
            if np.sum(bi[:, i]):
                y_left = i
                break

        for i in range(bi.shape[1] - 1, -1, -1):
            if np.sum(bi[:, i]):
                y_right = i
                break

        return x_low, x_top, y_left, y_right

    def get_simple_image(self, image):
        im = image.copy()
        li = self.get_object_labels(im)
        li = self.delete_small_obj(li)
        new_image = im
        new_image[li == 0] = 0
        for i in np.unique(li):
            if i == 0:
                continue
            w = li == i
            x_low, x_top, y_left, y_right = self.find_extr_points(w)

            color = image[w][0]
            new_image[x_low - 1:x_top + 1, y_left - 1:y_right + 1] = color

        return new_image




