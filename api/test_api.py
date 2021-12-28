from flask import Flask, jsonify
import requests
import time

main_dir_path = "/".join(os.getcwd().split("/")[:-1])

start_time = time.time()
url = 'http://0.0.0.0:5000/im_size'
my_img = {'image': open(main_dir_path + '/inputs/cropped_faces/Adele_crop.png', 'rb')}
r = requests.post(url, files=my_img, timeout=120)

print("--- {0} seconds ---".format(time.time() - start_time))
# convert server response into JSON format.
print(r.json())
