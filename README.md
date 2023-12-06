# HOR-classifier
method: A sliding window was created which iterated throughout the image and searched for the product in all the racks. It tries to match unique hash of the image and RGB color code and reads the Brand name on the label to classify the detected image

## components
Three components were used in this procedure to bring in the results, which are:
- Image hash similarity 
- RGB similarity
- Optical character recognition (OCR)

## working
The logic uses image hash, RGB values of the object and matches it with the YOLO detected window. the hash and RGB values of the window is then compared to the object values and a cosine similarity is dervied. on the top 3 similarity images, Optical character recogintion is performed for an extra layer of confirmation. Voila! we get the matched object.

## Author
Kaushal G / @kaushalg47


