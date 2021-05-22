import os
import sys
import traceback
import cv2
import numpy as np
try:
    from cores import Options,clean
    from util import util
    from util import image_processing as impro
    from models import loadmodel
except Exception as e:
    print(e)
    input('Please press any key to exit.\n')
    sys.exit(0)

# python server.py --gpu_id 0 --model_path ./pretrained_models/mosaic/clean_face_HD.pth
opt = Options()
opt.parser.add_argument('--port',type=int,default=4000, help='')
opt = opt.getparse(True)
netM = loadmodel.bisenet(opt,'mosaic')
netG = loadmodel.pix2pix(opt)

from flask import Flask, request
import base64
import shutil

app = Flask(__name__)

@app.route("/handle", methods=["POST"])
def handle():
    result = {}
    # to opencv img
    try:
        imgRec = request.form['img']
        imgByte = base64.b64decode(imgRec)
        img_np_arr = np.frombuffer(imgByte, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        result['img'] = imgRec
        result['info'] = 'readfailed'
        return result

    # run model
    try:
        if max(img.shape)>1080:
            img = impro.resize(img,720,interpolation=cv2.INTER_CUBIC)
        img = clean.cleanmosaic_img_server(opt,img,netG,netM)
    except Exception as e:
        result['img'] = imgRec
        result['info'] = 'procfailed'
        return result

    # return
    imgbytes = cv2.imencode('.jpg', img)[1]
    imgString = base64.b64encode(imgbytes).decode('utf-8')
    result['img'] = imgString
    result['info'] = 'ok'
    return result

app.run("0.0.0.0", port= opt.port, debug=opt.debug)