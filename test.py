import re
import requests
from PIL import Image

BASE_URL = 'https://api.flickr.com/services/rest/'
MAX_DISPLAYABLE_PHOTO_DIMENSION = 1600  # i.e. a photo with width and height <= 1600


def get_photo_info(photo_id, api_key):
    atts = {'api_key': api_key,
            'format': 'json',
            'method': 'flickr.photos.getInfo',
            'nojsoncallback': 1
            }
    atts['photo_id'] = photo_id
    photo = requests.get(BASE_URL, params=atts).json()['photo']
    return photo


def get_photo_sizes(photo_id, api_key):
    atts = {'api_key': api_key,
            'format': 'json',
            'method': 'flickr.photos.getSizes',
            'nojsoncallback': 1,
            'photo_id': photo_id
            }
    # assuming that :size attribute is always sorted, smallest to biggest
    data = requests.get(BASE_URL, params=atts).json()
    return data


def extract_photo_id(url):
    m = re.search("flickr.com/photos/\w+/(\d+)", url)
    if m:
        return m.groups()[0]
    else:
        # a
        return url

if __name__ == '__main__':

    # set API key
    api_key = "0e2a8ad6233af18575753b1bce914a26"

    url = "https://www.flickr.com/photos/zokuga/14392889220/"
    photo_id = extract_photo_id(url)
    photo_info = get_photo_info(photo_id, api_key)
    sizes_info = get_photo_sizes(photo_id, api_key)
    # print(photo_info)
    # print()
    url = sizes_info['sizes']['size'][0]['source']

    im = Image.open(requests.get(url, stream=True).raw)
    print(im)

