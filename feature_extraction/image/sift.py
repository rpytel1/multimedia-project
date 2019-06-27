import cv2


def extract_sift_feats(img):
    """
    Method to extract SIFT features
    :param img: provided image in BGR format
    :return:SIFT descriptors
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, dsc= sift.detectAndCompute(gray, None)
    return dsc
