import json
import time
import numpy as np


def process_category(category_data, user_data, post_data):
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

        # then uppdate the user data
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
    for data in tags_data:
        # first update the post data
        if data['Pid'] in post_data.keys():
            post_data[data['Pid']]['Title'] = data['Title']
            post_data[data['Pid']]['Mediatype'] = data['Mediatype']
            post_data[data['Pid']]['Alltags'] = data['Alltags'].split()
            post_data[data['Pid']]['Uid'] = data['Uid']
        else:
            post_data[data['Pid']] = {'Title': data['Title'],
                                      'Mediatype': data['Mediatype'],
                                      'Alltags': data['Alltags'].split(),
                                      'Uid': data['Uid']
                                      }

        # then uppdate the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['Title'] = data['Title']
                user_data[data['Uid']][data['Pid']]['Mediatype'] = data['Mediatype']
                user_data[data['Uid']][data['Pid']]['Alltags'] = data['Alltags'].split()
            else:
                user_data[data['Uid']][data['Pid']] = {'Title': data['Title'],
                                                       'Mediatype': data['Mediatype'],
                                                       'Alltags': data['Alltags'].split()
                                                       }
        else:
            user_data[data['Uid']] = {
                data['Pid']: {
                    'Title': data['Title'],
                    'Mediatype': data['Mediatype'],
                    'Alltags': data['Alltags'].split()
                }
            }
    return user_data, post_data


def process_geo(geo_data, user_data, post_data):
    for data in geo_data:
        # first update the post data
        if data['Pid'] in post_data.keys():
            post_data[data['Pid']]['Postdate'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['Postdate']))
            post_data[data['Pid']]['Longitude'] = np.NaN if not data['Longitude'] else float(data['Longitude'])
            post_data[data['Pid']]['Latitude'] = np.NaN if not data['Latitude'] else float(data['Latitude'])
            post_data[data['Pid']]['Geoaccuracy'] = int(data['Geoaccuracy'])
            post_data[data['Pid']]['Uid'] = data['Uid']
        else:
            post_data[data['Pid']] = {'Postdate': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['Postdate'])),
                                      'Longitude': np.NaN if not data['Longitude'] else float(data['Longitude']),
                                      'Latitude': np.NaN if not data['Latitude'] else float(data['Latitude']),
                                      'Geoaccuracy': int(data['Geoaccuracy']),
                                      'Uid': data['Uid']
                                      }

        # then uppdate the user data
        if data['Uid'] in user_data.keys():
            if data['Pid'] in user_data[data['Uid']].keys():
                user_data[data['Uid']][data['Pid']]['Postdate'] = time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                time.localtime(data['Postdate']))
                user_data[data['Uid']][data['Pid']]['Longitude'] = np.NaN if not data['Longitude'] \
                    else float(data['Longitude'])
                user_data[data['Uid']][data['Pid']]['Latitude'] = np.NaN if not data['Latitude'] \
                    else float(data['Latitude'])
                user_data[data['Uid']][data['Pid']]['Geoaccuracy'] = int(data['Geoaccuracy'])
            else:
                user_data[data['Uid']][data['Pid']] = {'Postdate': time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                 time.localtime(data['Postdate'])),
                                                       'Longitude': np.NaN if not data['Longitude']
                                                       else float(data['Longitude']),
                                                       'Latitude': np.NaN if not data['Latitude']
                                                       else float(data['Latitude']),
                                                       'Geoaccuracy': int(data['Geoaccuracy'])
                                                       }
        else:
            user_data[data['Uid']] = {
                data['Pid']: {
                    'Postdate': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['Postdate'])),
                    'Longitude': np.NaN if not data['Longitude'] else float(data['Longitude']),
                    'Latitude': np.NaN if not data['Latitude'] else float(data['Latitude']),
                    'Geoaccuracy': int(data['Geoaccuracy'])
                }
            }
    return user_data, post_data


if __name__ == '__main__':
    user_dataset = {}
    post_dataset = {}

    with open('data/train_all_json/train_category.json') as json_file:
        category_data = json.load(json_file)
    user_dataset, post_dataset = process_category(category_data, user_dataset, post_dataset)

    with open('data/train_all_json/train_tags.json') as json_file:
        tags_data = json.load(json_file)
    user_dataset, post_dataset = process_tags(category_data, user_dataset, post_dataset)

    with open('data/train_all_json/train_temporalspatial.json') as json_file:
        geo_data = json.load(json_file)
    user_dataset, post_dataset = process_geo(geo_data, user_dataset, post_dataset)

    with open('data/our_jsons/user_dataset.json', 'w') as outfile:
        json.dump(user_dataset, outfile)

    with open('data/our_jsons/post_dataset.json', 'w') as outfile:
        json.dump(post_dataset, outfile)