import cv2
import numpy as np

_CV2_BGR2HSV = cv2.COLOR_BGR2HSV


def compute_histogram(image, channels, bins, ranges):
    """
    universal function to extract histogram from certain image
    :param image: provided image
    :param channels: list of channels
    :param bins: number of bins per each channel
    :param ranges: what is the range of each of the channel values
    :return: histogram from provided image taking into account channels and bins
    """
    histogram = np.zeros(np.sum(bins))

    for i in range(0, len(channels)):
        channel = channels[i]
        channel_bins = bins[i]
        channel_range = ranges[i]
        channel_histogram = cv2.calcHist([image], [channel], None, [channel_bins], channel_range)

        start_index = int(np.sum(bins[0:channel]))
        end_index = start_index + channel_bins
        histogram[start_index:end_index] = channel_histogram.flatten()

    return histogram


def compute_hsv_histogram(image, bins_per_channel):
    """
    Function to extract HSV specific histogram
    :param image: provided image
    :param bins_per_channel: number of bins per channel
    :return:HSV features
    """
    hsv_image = cv2.cvtColor(image, _CV2_BGR2HSV)
    channels = [0, 1, 2]  # List of channels to analyze.
    ranges = [[0, 180], [0, 256], [0, 256]]  # Range per channel.
    return compute_histogram(hsv_image, channels, bins_per_channel, ranges)
