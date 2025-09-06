import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from rembg import remove


def grayscale_and_crop(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)


    # Convert the image to grayscale
    gray_img = img.convert('L')

    removed = remove(gray_img)

    img_np = np.array(removed)

    com = ndimage.center_of_mass(img_np)

    comx = com[1]
    comy = com[0]


    # Calculate the crop box
    crop_size = 1500  # Adjust the crop size as needed
    left = comx - crop_size // 2 + 700
    upper = comy - crop_size // 2 + 300
    right = comx + crop_size // 2 + 900
    lower = comy + crop_size // 2 + 350

    # Crop the image
    cropped_img = removed.crop((left, upper, right, lower))

    print(cropped_img.size)

    final = cropped_img.resize((130, 100))

    # Save the cropped image
    final.save(image_path.replace(".jpg", ".bmp"))

# Test the function
for i in range(1, 7):
    for j in range(1, 6):
        grayscale_and_crop(str(i) + "-" + str(j) + ".jpg")
# grayscale_and_crop(str(1) + "-" + str(5) + ".jpg")

