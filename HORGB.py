
import imagehash
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import easyocr
from difflib import SequenceMatcher
import cv2
from PIL import ImageEnhance
from PIL import ImageFilter
import easyocr
reader = easyocr.Reader(['en'])

# Load the image
image_path = r'' #shelf images set
image = Image.open(image_path)

obj_url = r"" #OBJECT URL
total_images = 0
matched_images = 0
def calculate_hash_similarity(window):
    image1 = window
    image2 = Image.open(obj_url)

    
    hash1 = imagehash.average_hash(image1)
    hash2 = imagehash.average_hash(image2)

    
    similarity = 1.0 - (hash1 - hash2) / len(hash1.hash) ** 2

    return similarity

def enhance(image_lr):
    desired_size = (image_lr.size[0] * 2, image_lr.size[1] * 2)  # Increase dimensions by a factor of 2
    
    # Perform bicubic interpolation to increase the resolution
    image_hr = image_lr.resize(desired_size, resample=Image.BICUBIC)
    
    # Display the high-resolution image
    return image_hr

def calculate_rgb_similarity(window):
    image1 = window
    image2 = Image.open(obj_url)

    
    image1 = image1.resize(image2.size)

    
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    
    histogram1 = image1.histogram()
    histogram2 = image2.histogram()

   
    similarity = sum(min(h1, h2) for h1, h2 in zip(histogram1, histogram2)) / float(sum(histogram1))
    return similarity





# Iterate over the image pixels with a sliding window
xywh_values = []
labels1 = []
for prediction in results['predictions']:   
    x, y, w, h, name= prediction['x'], prediction['y'], prediction['width'], prediction['height'], prediction['class']
    xywh_values.append((x, y, w, h))
    labels1.append(name)
    full_img = cv2.imread(r'') #shelf image
    
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
        # Extract the window region
        
    window = image.crop((x1, y1, x2, y2))
        

        # Calculate the hash of the window region
    hash_sim = calculate_hash_similarity(window)
    rgb_sim = calculate_rgb_similarity(window)
        

        
    total_images += 1
    img = window
    
    image_gray  = enhance(img)
    image_gray = image_gray.convert('L')
    plt.imshow(window)
    plt.axis('off')  
    plt.show()

        
    threshold = 128 
    image_binary = image_gray.point(lambda p: p > threshold and 255)

        
    image_bytes = io.BytesIO()
    image_binary.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    result = reader.readtext(image_bytes)
    print(result)

      
    text_detected = False
    target_text = "" #product name
    for text in result:
        if SequenceMatcher(None, target_text, text[1]).ratio() >= 0.5:
            text_detected = True
            break

    
            
    if hash_sim >= 0.1 and rgb_sim >= 0.1 and text_detected:   
        matched_images += 1
        plt.imshow(window)
        plt.axis('off')  
        plt.show()
        print(f"matched, hash:{hash_sim:.2%}, rgb:{rgb_sim:.2%}")
        x+=200
            
        
        

        #Images are not clear which is affecting the OCR to detect the text for classification
        