import json
import pickle
from json.decoder import JSONDecodeError
from feature_extraction.image.extract_img_feat import extract_feats_from_photo_metadata, extract_cv_feats
from feature_extraction.image.load_image_util import extract_photo_id, get_photo_info, get_photo_sizes, get_img


def get_img_feats(url):
    photo_id = extract_photo_id(url)
    photo_info = get_photo_info(photo_id)
    sizes_info = get_photo_sizes(photo_id)
    im = get_img(sizes_info)
    photo_metadata = extract_feats_from_photo_metadata(photo_info)
    photo_cv_feats = extract_cv_feats(im)
    photo_metadata.update(photo_cv_feats)
    return photo_metadata


if __name__ == '__main__':
    with open('data/our_jsons/user_dataset_updated.json') as json_file:
        user_data = json.load(json_file)

    final_dict = {}
    i = 0
    for key, value in user_data.items():
        user_value = {}
        for post_key, post_value in value.items():
            if post_value["Mediatype"] == 'photo':
                try:
                    post_value["img_feats"] = get_img_feats(post_value["image_url"])
                    user_value[post_key] = post_value
                except KeyError:
                    print("KeyError")
                except OSError:
                    print("OSError")
                except JSONDecodeError:
                    print("JSONDecodeError")
            print(i)
            i += 1
        final_dict[key] = user_value

    with open('data/our_jsons/user_dataset_computed.pickle', 'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
