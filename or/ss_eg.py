import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from PIL import Image
import numpy as np
import time


def main():

    # loading astronaut image
    img = skimage.data.astronaut()

    # load image
    img = np.asarray(Image.open("images/bedroom.jpg"))
    # img = np.asarray(Image.open("images/living_room.jpg"))
    ww = img.shape[0]  # width
    hh = img.shape[1]  # height
    all_size = hh * ww

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        # if r['size'] < all_size / 100:
        #     continue
        # distorted rects
        x, y, w, h = r['rect']
        # too narrow or short
        if w < ww / 100 or h < hh / 100:
            continue
        # eliminate too small or too big
        if r['size'] < all_size / 500.0 or r['size'] > all_size / 3:
            continue
        # elimite small distorted rects
        if (w / h > 1.2 or h / w > 1.2) and r['size'] < all_size / 100.0:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    for x, y, w, h in candidates:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plt.ion()
        ax.imshow(img)
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.imshow(img)
        ax.add_patch(rect)
        # plt.show()
        plt.show()
        plt.pause(0.5)

    plt.close('all')


if __name__ == "__main__":
    main()
