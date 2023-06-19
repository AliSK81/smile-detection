import cv2

from image_util import ImageUtil

if __name__ == '__main__':
    img = ImageUtil.load_image('resources/genki4k/files/file0001.jpg', mode=cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = ImageUtil.resize_image(img, size=(128, 128))
    print(img.shape)
