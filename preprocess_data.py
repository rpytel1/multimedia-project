import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import json


def make_category(data):
    data_lst = [[pid, data[pid]['Category'], data[pid]['Concept'], data[pid]['Subcategory']] for pid in data.keys()]
    df = pd.DataFrame(data_lst, columns=['Pid', 'Category', 'Concept', 'Subcategory'])
    pids = df['Pid']
    df.drop('Pid', axis=1, inplace=True)
    df = df.apply(lambda col: pd.factorize(col)[0])
    df['Pid'] = pids
    return df


def make_tags(data):
    titles = [data[pid]['Title'] for pid in data.keys()]
    title_vectorizer = CountVectorizer(decode_error='ignore').fit_transform(titles)
    title_matrix = title_vectorizer.toarray()

    tags = [' '.join(data[pid]['Alltags']) for pid in data.keys()]
    tag_vectorizer = CountVectorizer(decode_error='ignore').fit_transform(tags)
    tag_matrix = tag_vectorizer.toarray()

    pids = np.asarray(list(data.keys()))
    # title_tags = np.concatenate((title_matrix, tag_matrix), axis=1)
    return pd.DataFrame(np.concatenate((pids, tag_matrix), axis=1))


def make_dates(data):
    data_lst = [[pid, data[pid]['Postdate']] for pid in data.keys()]
    df = pd.DataFrame(data_lst, columns=['Pid', 'Postdate'])
    df['Postdate'] = pd.to_datetime(df['Postdate'], format='%Y-%m-%d %H:%M:%S')
    return df


def split_data(data, usr_data):
    final_dict = {}
    for user in usr_data.keys():
        posts = data.loc[list(usr_data[user].keys())]
        posts.sort_values('Postdate', inplace=True)
        final_dict[user] = {'train_set': posts.iloc[:int(posts.shape[0]/2)]}
        final_dict[user]['test_set'] = posts.iloc[int(posts.shape[0]/2):]
    return final_dict


if __name__ == '__main__':
    with open('data/our_jsons/user_dataset_updated.json') as json_file:
        user_data = json.load(json_file)

    # split data to feature categories
    category_dict = {}
    tags_dict = {}
    # image_dict = {} # TODO: implement it when image features are available
    dates_dict = {}
    for user in user_data.keys():
        for pid, vals in user_data[user].items():
            for feat_type in vals.keys():
                if feat_type in ['Category', 'Concept', 'Subcategory']:
                    if pid not in category_dict.keys():
                        category_dict[pid] = {feat_type: vals[feat_type]}
                    else:
                        category_dict[pid][feat_type] = vals[feat_type]
                elif feat_type in ['Title', 'Alltags']:
                    if pid not in tags_dict.keys():
                        tags_dict[pid] = {feat_type: vals[feat_type]}
                    else:
                        tags_dict[pid][feat_type] = vals[feat_type]
                # elif feat_type == 'img_feats':
                #     image_dict[pid][feat_type] = vals[feat_type]
                elif feat_type == 'Postdate':
                    if pid not in dates_dict.keys():
                        dates_dict[pid] = {feat_type: vals[feat_type]}
                    else:
                        dates_dict[pid][feat_type] = vals[feat_type]

    category_data = make_category(category_dict)
    # tags_data = make_tags(tags_dict)  #TODO: fix problem with dimensions
    dates_data = make_dates(dates_dict)

    all_data = pd.merge(dates_data, category_data, on="Pid")
    # all_data = pd.concat([pd.merge(dates_data, category_data, on="Pid"), tags_data.drop([0], axis=1)], axis=1)
    all_data = all_data.set_index('Pid')

    final_data = split_data(all_data, user_data)

    print('preprocessing completed!!!')
