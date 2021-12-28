from flask import Flask, jsonify
import requests
import time

start_time = time.time()
url = 'http://0.0.0.0:5000/im_size'
my_img = {'image': open('/Users/nyt21/Devel/GFPGAN/GFPGAN/inputs/cropped_faces/Adele_crop.png', 'rb')}
r = requests.post(url, files=my_img, timeout=120)

print("--- {0} seconds ---".format(time.time() - start_time))
# convert server response into JSON format.
print(r.json())
