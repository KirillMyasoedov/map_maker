import os
from PIL import Image
import numpy as np


class AccuracyCalculator(object):
    def __init__(self):
        self.positive_predicted_mask = None
        self.negative_predicted_mask = None
        self.positive_true_mask = None
        self.negative_true_mask = None

    def calculate_image_accuracy(self, true, predicted):
        self.positive_predicted_mask = predicted == 100
        self.negative_predicted_mask = predicted == 0
        self.positive_true_mask = (true[:, :, 0] == 255) * (true[:, :, 1] == 255) * (true[:, :, 2] == 255)
        self.negative_true_mask = ~self.positive_true_mask

        TP = (self.positive_predicted_mask * self.positive_true_mask).sum()
        TN = (self.negative_predicted_mask * self.negative_true_mask).sum()
        FP = (self.positive_predicted_mask * self.negative_true_mask).sum()
        FN = (self.negative_predicted_mask * self.positive_true_mask).sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_score

    def save_comparison(self, name, path):
        image = np.zeros((self.positive_true_mask.shape[0], self.negative_true_mask.shape[1], 3), dtype=np.uint8)
        image[self.positive_true_mask * self.positive_predicted_mask] = [0, 0, 255]
        image[self.positive_true_mask * self.negative_predicted_mask] = [255, 0, 0]
        image[self.negative_true_mask * self.positive_predicted_mask] = [0, 255, 0]

        image = Image.fromarray(image)
        image.save(os.path.join(path, name + '.png'))
