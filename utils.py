import os
import settings
import cv2
import numpy as np
from imutils.perspective import four_point_transform


def save_upload_image(fileObj):
    filename = fileObj.filename    
    name , ext = filename.split('.')    
    save_filename = 'upload.'+ext
    upload_image_path = settings.join_path(settings.SAVE_DIR, 
                                           save_filename)
    
    fileObj.save(upload_image_path)
    
    return upload_image_path


def array_to_json_format(numpy_array):
    points = []
    for pt in numpy_array.tolist():
        points.append({'x':pt[0],'y':pt[1]})
        
    return points

class DocumentScan():
    def __init__(self):
        pass
    
    @staticmethod
    def resizer(image, width=500):
        # get width and height
        h, w, c = image.shape
        
        height = int((h / w) * width)
        size = (width, height)
        image = cv2.resize(image, (width, height))
        return image, size
    
    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
    
    def document_scanner(self, image_path):
        self.image = cv2.imread(image_path)
        img_re, self.size = self.resizer(self.image)
        filename = 'resize_image.jpg'
        RESIZE_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR, filename)
        
        cv2.imwrite(RESIZE_IMAGE_PATH, img_re)
        
        try:
            # Code de traitement d'image ici (sans prédiction de coordonnées du document)
                
            return None, self.size
        except:
            return None, self.size
    
    
    def calibrate_to_original_size(self, four_points):
        # find four points for original image
        
        multiplier = self.image.shape[1] / self.size[0]
        four_points_orig = four_points * multiplier
        four_points_orig = four_points_orig.astype(int)
        wrap_image = four_point_transform(self.image, four_points_orig)
        # apply magic color to wrap image
        magic_color = self.apply_brightness_contrast(wrap_image, brightness=40, contrast=60)
        
        return magic_color
