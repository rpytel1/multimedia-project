import cv2


def extract_sift_feats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, dsc= sift.detectAndCompute(gray, None)
    return dsc
