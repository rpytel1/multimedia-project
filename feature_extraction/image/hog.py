import cv2

WIDTH = 25
HEIGHT = 25


def create_hog_descriptor():
    """
    Method creating configured HOG descriptor
    :return:configured HOG descriptor
    """
    win_size = (24, 24)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = 4.
    histogram_norm_type = 0
    l2_hys_threshold = 2.0000000000000001e-01
    gamma_correction = 0
    nlevels = 64
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                             histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)


def resize_img(img):
    """
    Method resizing image to designated dimensions. Currently it is 25x25
    :param img: provided image in original dimensions
    :return: resized image
    """
    dim = (WIDTH, HEIGHT)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def extract_hog_feats(img):
    """
    Method extracting HOG features for certain image
    :param img: provided image
    :return:computed HOG features
    """
    hog_descriptor = create_hog_descriptor()
    img = resize_img(img)
    return hog_descriptor.compute(img)
