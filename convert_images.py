import os

from PIL import Image

directory = '/Users/hannes/PetImages/'
directory_suffix = 'training_set/'
folders = ['Cat/', 'Dog/']
color_schema = 'L'  # L - grayscale
resized_px = 200

if not os.path.exists(directory + directory_suffix):
    os.makedirs(directory + directory_suffix)

for folder in folders:

    _initial_path = directory + folder
    _final_path = directory + directory_suffix + folder
    _files = [f for f in os.listdir(_initial_path) if os.path.isfile(os.path.join(_initial_path, f))]

    if not os.path.exists(_final_path):
        os.makedirs(_final_path)

    for f in _files:
        try:
            image = Image.open(_initial_path + f)
        except IOError:
            continue
        y, x = image.size
        y = x if x > y else y
        resized_image = Image.new(color_schema, (y, y), (255, ))
        try:
            resized_image.paste(image, image.getbbox())
        except ValueError:
            continue
        resized_image = resized_image.resize((resized_px, resized_px), Image.ANTIALIAS)
        resized_image.save(_final_path + f, 'jpeg', quality=90)
