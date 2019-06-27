import json
import time
import numpy as np


def process_category(category_data, user_data, post_data):
    """
    Function that takes the original category data as input and updates the posts and user data structures
    :param category_data: the category data in the original dataset
    :param user_data: the user data structure
    :param post_data: the post data structure
    :return: the updated structures
    """
    for data in category_data:
        # first update the post data
        if data['Pid'] in post_data.keys():
            post_data[data['Pid']]['Category'] = data['Category']
            post_data[data['Pid']]['Concept'] = data['Concept']
            post_data[data['Pid']]['Subcategory'] = data['Subcategory']
            post_data[data['Pid']]['Uid'] = data['Uid']
        else:
            post_data[data['Pid']] = {'Category': data['Category'],
                                      'Concept': data['Concept'],
                                      'Subcategory': data['Subcategory'],
                                      'Uid': data['Uid']
                                      }

        # then update the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['Category'] = data['Category']
                user_data[data['Uid']][data['Pid']]['Concept'] = data['Concept']
                user_data[data['Uid']][data['Pid']]['Subcategory'] = data['Subcategory']
            else:
                user_data[data['Uid']][data['Pid']] = {'Category': data['Category'],
                                                       'Concept': data['Concept'],
                                                       'Subcategory': data['Subcategory']
                                                       }
        else:
            user_data[data['Uid']] = {
                data['Pid']: {
                    'Category': data['Category'],
                    'Concept': data['Concept'],
                    'Subcategory': data['Subcategory']
                }
            }
    return user_data, post_data


def process_tags(tags_data, user_data, post_data):
    """
    Function that takes the original tag data as input and updates the posts and user data structures
    :param tags_data: the tag data in the original dataset
    :param user_data: the user data structure
    :param post_data: the post data structure
    :return: the updated structures
    """
    for data in tags_data:
        # first update the post data
        if data['Pid'] in post_data.keys():
            post_data[data['Pid']]['Title'] = data['Title']
            post_data[data['Pid']]['Mediatype'] = data['Mediatype']
            post_data[data['Pid']]['Alltags'] = data['Alltags'].split()  # turn string into list
            post_data[data['Pid']]['Uid'] = data['Uid']
        else:
            post_data[data['Pid']] = {'Title': data['Title'],
                                      'Mediatype': data['Mediatype'],
                                      'Alltags': data['Alltags'].split(),  # turn string into list
                                      'Uid': data['Uid']
                                      }

        # then update the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['Title'] = data['Title']
                user_data[data['Uid']][data['Pid']]['Mediatype'] = data['Mediatype']
                user_data[data['Uid']][data['Pid']]['Alltags'] = data['Alltags'].split()  # turn string into list
            else:
                user_data[data['Uid']][data['Pid']] = {'Title': data['Title'],
                                                       'Mediatype': data['Mediatype'],
                                                       'Alltags': data['Alltags'].split()  # turn string into list
                                                       }
        else:
            user_data[data['Uid']] = {
                data['Pid']: {
                    'Title': data['Title'],
                    'Mediatype': data['Mediatype'],
                    'Alltags': data['Alltags'].split()  # turn string into list
                }
            }
    return user_data, post_data


def process_geo(geo_data, user_data, post_data):
    """
    Function that takes the original geo-location data as input and updates the posts and user data structures
    :param geo_data: the geo-location  data in the original dataset
    :param user_data: the user data structure
    :param post_data: the post data structure
    :return: the updated structures
    """
    for data in geo_data:
        # first update the post data
        if data['Pid'] in post_data.keys():
            # preprocess the postdate in the appropriate format
            post_data[data['Pid']]['Postdate'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(data['Postdate'])))
            post_data[data['Pid']]['Longitude'] = np.NaN if not data['Longitude'] else float(data['Longitude'])
            post_data[data['Pid']]['Latitude'] = np.NaN if not data['Latitude'] else float(data['Latitude'])
            post_data[data['Pid']]['Geoaccuracy'] = int(data['Geoaccuracy'])
            post_data[data['Pid']]['Uid'] = data['Uid']
        else:
            post_data[data['Pid']] = {'Postdate': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(data['Postdate']))),
                                      'Longitude': np.NaN if not data['Longitude'] else float(data['Longitude']),
                                      'Latitude': np.NaN if not data['Latitude'] else float(data['Latitude']),
                                      'Geoaccuracy': int(data['Geoaccuracy']),
                                      'Uid': data['Uid']
                                      }

        # then update the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['Postdate'] = time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                time.localtime(int(data['Postdate'])))
                user_data[data['Uid']][data['Pid']]['Longitude'] = np.NaN if not data['Longitude'] \
                    else float(data['Longitude'])
                user_data[data['Uid']][data['Pid']]['Latitude'] = np.NaN if not data['Latitude'] \
                    else float(data['Latitude'])
                user_data[data['Uid']][data['Pid']]['Geoaccuracy'] = int(data['Geoaccuracy'])
            else:
                user_data[data['Uid']][data['Pid']] = {'Postdate': time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                 time.localtime(int(data['Postdate']))),
                                                       'Longitude': np.NaN if not data['Longitude']
                                                       else float(data['Longitude']),
                                                       'Latitude': np.NaN if not data['Latitude']
                                                       else float(data['Latitude']),
                                                       'Geoaccuracy': int(data['Geoaccuracy'])
                                                       }
        else:
            user_data[data['Uid']] = {
                data['Pid']: {
                    'Postdate': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(data['Postdate']))),
                    'Longitude': np.NaN if not data['Longitude'] else float(data['Longitude']),
                    'Latitude': np.NaN if not data['Latitude'] else float(data['Latitude']),
                    'Geoaccuracy': int(data['Geoaccuracy'])
                }
            }
    return user_data, post_data


def process_image_url(image_urls, user_data, post_data, helper_data):
    """
    Function that takes the original image urls as input and updates the posts and user data structures
    :param image_urls: the image urls in the original dataset
    :param user_data: the user data structure
    :param post_data: the post data structure
    :param helper_data: helper data structure wo assist with the user id and post id inconsistencies in the
    train_img.txt file
    :return: the updated structures
    """
    for ind, data in enumerate(helper_data):
        # first update the post data
        if data['Pid'] in post_data.keys():
            post_data[data['Pid']]['image_url'] = image_urls[ind]
        else:
            post_data[data['Pid']] = {'image_url': image_urls[ind]}

        # then update the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['image_url'] = image_urls[ind]
            else:
                user_data[data['Uid']][data['Pid']] = {'image_url': image_urls[ind]}
        else:
            user_data[data['Uid']] = {
                data['Pid']: {'image_url': image_urls[ind]}
            }
    return user_data, post_data


if __name__ == '__main__':
    user_dataset = {}
    post_dataset = {}

    # update the above data structure for each feature category
    with open('data/train_all_json/train_category.json') as json_file:
        category_data = json.load(json_file)
    user_dataset, post_dataset = process_category(category_data, user_dataset, post_dataset)
    print('Category features added')

    with open('data/train_all_json/train_tags.json') as json_file:
        tags_data = json.load(json_file)
    user_dataset, post_dataset = process_tags(tags_data, user_dataset, post_dataset)
    print('Tag features added')

    with open('data/train_all_json/train_temporalspatial.json') as json_file:
        geo_data = json.load(json_file)
    user_dataset, post_dataset = process_geo(geo_data, user_dataset, post_dataset)
    print('Temporal-Spatial features added')

    with open('data/train_all_json/train_img.txt') as f:
        image_urls = f.readlines()
    image_urls = [x.strip() for x in image_urls]
    user_dataset, post_dataset = process_image_url(image_urls, user_dataset, post_dataset, category_data)
    print('Image urls added')

    # save the created structures
    with open('data/our_jsons/user_dataset.json', 'w') as outfile:
        json.dump(user_dataset, outfile)

    with open('data/our_jsons/post_dataset.json', 'w') as outfile:
        json.dump(post_dataset, outfile)