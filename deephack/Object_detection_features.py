import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label


class ObjectDetectionFeatures:
    def __init__(self, env):
        self.env = env
        self.act = env.action_space
        self.proba_computed=False

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

    def get_object_labels(self, image):
        bin_im = self.binarization(image)

        labeled_im = label(bin_im)
        for i in np.unique(labeled_im):
            w = labeled_im == i
            if np.sum(w) < 3:
                labeled_im[w] = 0

        #plt.imshow(labeled_im)
        #plt.show()
        return labeled_im

    def find_all_obj_classes(self, n_starts=10, max_samples_im_count=5000):
        n_samples = 0
        last_changes = n_samples
        classes = set([])

        for i in range(n_starts):
            self.env.reset()
            while not self.env.ale.game_over():
                n_cl = len(classes)
                if n_samples % 1000 == 0:
                    print n_samples
                n_samples += 1
                image = self.env.ale.getScreenGrayscale()
                li = self.get_object_labels(image)
                for l in np.unique(li):
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

    

