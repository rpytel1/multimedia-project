from feature_extraction.image.histogram import compute_hsv_histogram
import cv2

from feature_extraction.image.hog import extract_hog_feats


def extract_feats_from_photo_metadata(photo_metadata):
    metadata_feats = {}
    metadata_feats['id'] = photo_metadata['id']
    metadata_feats['date_uploaded'] = int(photo_metadata['dateuploaded'])
    metadata_feats['owner_id'] = photo_metadata['owner']['nsid']
    metadata_feats['views'] = int(photo_metadata['views'])
    metadata_feats['comments'] = int(photo_metadata['comments']['_content'])
    return metadata_feats


def extract_cv_feats(photo):
    return True


def extract_hsv_histogram(photo):
    hsv_histogram = {"hsv_hist": compute_hsv_histogram(photo, [8, 8, 8])}
    return hsv_histogram


def extract_hog_features(photo):
    hog = {"hog": extract_hog_feats(photo)}
    return hog


def extract_vgg_feats():
    pass
