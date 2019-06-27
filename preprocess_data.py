import pandas as pd
import pickle
import gensim.downloader as api
import nltk
import numpy as np

model = api.load("glove-wiki-gigaword-100")
print(model['man'])
tokenizer = nltk.RegexpTokenizer(r'\w+')


def apply_one_hot(df, categorical_features):
    """
    Function that applies one-hot encoding to the features specified in the categorical_features list (eventually
    not used)
    :param df: the dataframe of the dataset
    :param categorical_features: the features on which one-hot will be applied
    :return: the updated dataframe
    """
    for col in categorical_features:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def make_category(data):
    """
    Function for preprocessing the category related data by assigning numerical values to their categorical ones
    :param data: the initial category data
    :return: the converted to numerical representation category data
    """
    data_lst = [[pid, data[pid]['Category'], data[pid]['Concept'], data[pid]['Subcategory']] for pid in data.keys()]
    df = pd.DataFrame(data_lst, columns=['Pid', 'Category', 'Concept', 'Subcategory'])
    pids = df['Pid']
    df.drop('Pid', axis=1, inplace=True)
    df = df.apply(lambda col: pd.factorize(col)[0])  # assign the numerical values
    df['Pid'] = pids
    return df


def embedding(title):
    """
    TODO: This is for you Rafal
    :param title:
    :return:
    """
    words = tokenizer.tokenize(title)
    final_emb = np.zeros((1, 100))
    i = 1
    for word in words:
        try:
            final_emb += np.array(model[word])
            i += 1
        except:
            ()
    return final_emb/i


def make_tags(data):
    """
    Function for preprocessing the tag data by applying the glove embedding on the title and the tags of each
    post
    :param data: the tag data
    :return: the updated dataframe
    """
    titles = [[pid] + [i for sub in embedding(data[pid]['Title']).T.tolist() for i in sub] + [i for sub in embedding(
        " ".join(data[pid]['Alltags'])).T.tolist() for i in sub] for pid in data.keys()]

    headers = ['Pid'] + ['title_emb_' + str(i) for i in range(100)] + ['tags_emb_' + str(i) for i in range(100)]
    df = pd.DataFrame(titles, columns=headers)

    return df


def make_dates(data):
    """
    Function for preprocessing the dates in the dataset into the appropriate dataframe format
    :param data: the initial postdate data
    :return: the generated dataframe
    """
    data_lst = [[pid, data[pid]['Postdate']] for pid in data.keys()]
    df = pd.DataFrame(data_lst, columns=['Pid', 'Postdate'])
    df['Postdate'] = pd.to_datetime(df['Postdate'], format='%Y-%m-%d %H:%M:%S')
    return df


def make_image_fts(data):
    """
    Function for preprocessing the two types of image features extracted from the images in the dataset
    :param data: the image features
    :return: the created dataframe
    """
    data_lst = [[pid] + data[pid]['hsv_hist'] + [i for sub in data[pid]['hog'] for i in sub] for pid in data.keys()]
    headers = ['Pid'] + ['hsv_hist_' + str(i) for i in range(24)] + ['hog_' + str(i) for i in range(144)]
    df = pd.DataFrame(data_lst, columns=headers)
    return df


def split_data(data, usr_data):
    """
    Function for splitting the posts for each user into a training and a test set according to their postdate
    :param data: the dataset in a dataframe format with the post id as the index
    :param usr_data: the original data with user ids as the keys and the posts as the values
    :return: the split dataset for each user
    """
    final_dict = {}
    for user in usr_data.keys():
        posts = data.loc[list(usr_data[user].keys())]  # locate the posts of each user
        posts.sort_values('Postdate', inplace=True)  # sort values according to postdate
        final_dict[user] = {'train_set': posts.iloc[:int(posts.shape[0] / 2)]}  # keep half of the posts as
        final_dict[user]['test_set'] = posts.iloc[int(posts.shape[0] / 2):]  # training and half as test set
    return final_dict


if __name__ == '__main__':
    with open("data/our_jsons/user_dataset_computed.pickle", "rb") as input_file:
        complete_data = pickle.load(input_file)

    # split data to feature categories and preprocess each feature category
    category_dict = {}
    tags_dict = {}
    image_dict = {}
    dates_dict = {}
    for user in complete_data.keys():
        for pid, vals in complete_data[user].items():
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
                elif feat_type == 'img_feats':
                    image_dict[pid] = {'hsv_hist': vals[feat_type]['hsv_hist'].tolist(),
                                       'hog': vals[feat_type]['hog'].tolist()}
                elif feat_type == 'Postdate':
                    if pid not in dates_dict.keys():
                        dates_dict[pid] = {feat_type: vals[feat_type]}
                    else:
                        dates_dict[pid][feat_type] = vals[feat_type]

    category_data = make_category(category_dict)
    tags_data = make_tags(tags_dict)
    dates_data = make_dates(dates_dict)
    image_data = make_image_fts(image_dict)

    # merge all feature dataframes
    all_data = pd.merge(pd.merge(pd.merge(dates_data, category_data, on="Pid"), image_data, on="Pid"), tags_data,
                        on="Pid")
    all_data = all_data.set_index('Pid')  # and set the post id as the index

    # created the split dataset per user
    final_data = split_data(all_data, complete_data)
    with open('data/our_jsons/final_dataset_with_tags.pickle', 'wb') as handle:
        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('preprocessing completed!!!')
