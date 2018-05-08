import cv2
import numpy as np
from PIL import Image
import os
import pytesseract

class preProcessing:

    def pre_proc_image(self,img):
        img = np.array(img,dtype=np.uint8)
        retvalue, img = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold Changes",np.array(img,dtype=np.uint8))
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
    image = cv2.imread('captcha3.png')
    image = Image.fromarray(image)
    p = preProcessing()
    imgOP = p.pre_proc_image(image)

    #Erosion of Image
    imageEr = cv2.imread("invCaptcha1.png")
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(imageEr, kernel, iterations=2)
    # cv2.imshow("OpenImage", erosion)
    # cv2.waitKey(0)

    filename = "Output1.png".format(os.getpid())
    cv2.imwrite(filename, imgOP)

    text = pytesseract.image_to_string(Image.open(filename),config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKMNOPQRSTUVWXYZ")

    #os.remove(filename)
    print(text)