import json
import pickle

from feature_extraction.image.extract_img_feat import extract_feats_from_photo_metadata, extract_cv_feats
from feature_extraction.image.load_image_util import extract_photo_id, get_photo_info, get_photo_sizes, \
    get_img

if __name__ == '__main__':
    with open("data/train_all_json/train_img.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    print(len(content))
    dict_list =[]
    for url in content[0:5]:
        photo_id = extract_photo_id(url)
        photo_info = get_photo_info(photo_id)
        sizes_info = get_photo_sizes(photo_id)
        im = get_img(sizes_info)
        photo_metadata = extract_feats_from_photo_metadata(photo_info)
        photo_cv_feats = extract_cv_feats(im)
        photo_metadata.update(photo_cv_feats)
        dict_list.append(photo_metadata)

    with open('data/img_feats/data.pickle', 'wb') as handle:
        pickle.dump(dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
