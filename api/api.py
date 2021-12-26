from flask import Flask, request, jsonify
import cv2
from PIL import Image
import utilx_api as utilx_api
import numpy as np
from basicsr.utils import imwrite

conf, restorer = utilx_api.prepare_data()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def ping():
    """ Check if the API is up and running. """
    return {"running": True}

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    input_img_orig = Image.open(file.stream)
    input_img = cv2.cvtColor(np.array(input_img_orig), cv2.COLOR_RGB2BGR)

    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=conf['aligned'], only_center_face=conf['only_center_face'], paste_back=conf['paste_back'])

    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        imwrite(restored_face, '/Users/nyt21/Devel/GFPGAN/GFPGAN/results/myresult.png')
    # output_image.save('/Users/nyt21/Devel/GFPGAN/GFPGAN/results/output_ai.png')


    return jsonify({'msg': 'success', 'size': [input_img_orig.width, input_img_orig.height]})


if __name__ == "__main__":
    app.run(debug=True)