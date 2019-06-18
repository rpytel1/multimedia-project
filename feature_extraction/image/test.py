from feature_extraction.image.load_image_util import extract_photo_id, get_photo_info, get_photo_sizes, \
    get_img

if __name__ == '__main__':
    # set API key

    url = "https://www.flickr.com/photos/35797910@N08/3544891702"
    photo_id = extract_photo_id(url)
    photo_info = get_photo_info(photo_id)
    sizes_info = get_photo_sizes(photo_id)
    im = get_img(sizes_info)

    print(im)
