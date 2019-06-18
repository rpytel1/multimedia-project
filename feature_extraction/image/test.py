from feature_extraction.image.extract_img_feat import extract_feats_from_photo_metadata, extract_cv_feats
from feature_extraction.image.load_image_util import extract_photo_id, get_photo_info, get_photo_sizes, \
    get_img

if __name__ == '__main__':
    url = "https://www.flickr.com/photos/35797910@N08/3544891702"
    photo_id = extract_photo_id(url)
    photo_info = get_photo_info(photo_id)
    sizes_info = get_photo_sizes(photo_id)
    im = get_img(sizes_info)
    photo_metadata = extract_feats_from_photo_metadata(photo_info)
    photo_cv_feats = extract_cv_feats(im)
