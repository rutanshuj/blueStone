import cv2
import numpy as np
from PIL import Image
import os
import pytesseract

class preProcessing:

    def pre_proc_image(self,img):
        img = np.array(img,dtype=np.uint8)
        retvalue, img = cv2.threshold(img, 4, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold Changes",img)
        # cv2.waitKey(0)
        img = Image.fromarray(img)
        img_removed_noise = self.apply_median_filter(img)
        return img_removed_noise

    def apply_median_filter(self, img):
        img_gray = img.convert('L')
        img_gray = cv2.medianBlur(np.asarray(img_gray), 3)
        img_bw = (img_gray > np.mean(img_gray)) * 255
        return img_bw

if __name__ == '__main__':
    image = cv2.imread('captcha.png')
    image = Image.fromarray(image)
    p = preProcessing()
    imgOP = p.pre_proc_image(image)

    filename = "Output.png".format(os.getpid())
    cv2.imwrite(filename, np.array(imgOP))

    text = pytesseract.image_to_string(Image.open(filename))
    #os.remove(filename)
    print(text)