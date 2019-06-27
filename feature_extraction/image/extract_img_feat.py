from feature_extraction.image.histogram import compute_hsv_histogram
from feature_extraction.image.hog import extract_hog_feats
from feature_extraction.image.sift import extract_sift_feats


def extract_feats_from_photo_metadata(photo_metadata):
    """
    Function to extract metadata information from json returned from Flickr platform
    :param photo_metadata: json returned from Flickr platform
    :return:dictionary of metadata information about image
    """
    metadata_feats = {'id': photo_metadata['id'], 'date_uploaded': int(photo_metadata['dateuploaded']),
                      'owner_id': photo_metadata['owner']['nsid'], 'views': int(photo_metadata['views']),
                      'comments': int(photo_metadata['comments']['_content'])}
    return metadata_feats


def extract_cv_feats(photo, with_sift=False):
    """
    Aggregate function to extract HSV, HOG and possibly SIFT features for certain image
    :param photo: provided photo
    :param with_sift: boolean flag enabling SIFT features extraction
    :return: dictionary of computer vision image features
    """
    feats = {"hsv_hist": extract_hsv_histogram(photo), "hog": extract_hog_features(photo)}
    if with_sift:
        feats["sift"] = extract_sift_features(photo)
    return feats


def extract_hsv_histogram(photo):
    return compute_hsv_histogram(photo, [8, 8, 8])


def extract_hog_features(photo):
    return extract_hog_feats(photo)


def extract_sift_features(photo):
    return extract_sift_feats(photo)
