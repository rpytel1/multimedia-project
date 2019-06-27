import cv2
import requests
from PIL import Image
import numpy as np
from io import BytesIO

BASE_URL = 'https://api.flickr.com/services/rest/'
MAX_DISPLAYABLE_PHOTO_DIMENSION = 1600  # i.e. a photo with width and height <= 1600
API_KEY = "0e2a8ad6233af18575753b1bce914a26"
API_KEY2 = "9c5e293c8f652b701785ba4dc2ecd5d0"


def get_photo_info(photo_id):
    """
    Function making request to Flickr API in order to retrieve metadata of the photo
    :param photo_id: id of the photo, extracted from the link
    :return:response from the Flickr API
    """
    atts = {'api_key': API_KEY, 'format': 'json', 'method': 'flickr.photos.getInfo', 'nojsoncallback': 1,
            'photo_id': photo_id}
    photo = requests.get(BASE_URL, params=atts).json()['photo']
    return photo


def get_photo_sizes(photo_id):
    """
    Function making request to Flickr API in order to retrieve available photo sizes of a certain photo on Flickr
    :param photo_id:id of the photo to retrieve its available sizes
    :return:json response from Flickr API
    """
    atts = {'api_key': API_KEY2,
            'format': 'json',
            'method': 'flickr.photos.getSizes',
            'nojsoncallback': 1,
            'photo_id': photo_id
            }
    # assuming that :size attribute is always sorted, smallest to biggest
    data = requests.get(BASE_URL, params=atts).json()
    return data


def extract_photo_id(url):
    """
    Function to extract photo id from link provided
    :param url: original link to the photo page on Flickr
    :return:photo id
    """
    m = url.split("/")
    if m:
        return m[5]
    else:
        return url


def get_img(sizes_info):
    """
    Function to get image from Flickr platform and transform it to BGR format.
    :param sizes_info: json list of available sizes of an image
    :return:image from Flickr
    """
    url = sizes_info['sizes']['size'][0]['source']
    # im = np.asarray(Image.open(requests.get(url, stream=True).raw))
    im = np.asarray(Image.open(BytesIO((requests.get(url, stream=True)).content)))
    if len(im.shape) is 3:
        imcv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        imcv = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return imcv