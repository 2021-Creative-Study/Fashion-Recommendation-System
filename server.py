import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import json
import os


app = Flask(__name__)

fe = FeatureExtractor()

@app.route('/test', methods=['GET', 'POST'])
def test():
    li = []
    li2 = []
    li3 = []
    for i in range(6):
        li.append('static/test_new_all/0a8z8t06iq.jpg')
    for i in range(6):
        li2.append('static/test_new_all/0a8z8t06iq.jpg')
    for i in range(6):
        li3.append('static/test_new_all/0a8z8t06iq.jpg')
    return render_template('test.html',li=li,li2=li2,li3=li3)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        # Save query image
        image = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        image.save(uploaded_img_path)

        # Run search
        query = fe.extract(uploaded_img_path)
        # query2 = fe.extract2(경로)

        imglabel = query[0]
        imgpath = query[1]
        scores = query[2]
        imgpath2 = query[3]
        classes = ['top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top', 
        'hood', 'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants', 
        'coat', 'jacket', 'jumper', 'padding jacket', 'best', 'kadigan', 
        'zip up', 'dress', 'jumpsuit']
        name = classes[imglabel-1]
        
        print('finish')

        #---------추가---------
        img_li = []
        img_final = []
        img_path = './static/test_new_all'
        img_name = './static/test_public.json'
        cnt = 0
        with open(img_name, 'r') as f:
            json_data = json.load(f)
        for i in range(len(json_data['images'])):
            data = json.dumps(json_data['images'][i]['file_name'])
            data = data.replace('"','')
            img_li.append(data)
            query = fe.extract2(os.path.join(img_path, img_li[i]))
            if query==imglabel:
                cnt+=1
                if cnt == 13:
                    break
                img_final.append(os.path.join(img_path, img_li[i]))

            

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               imglabel = imglabel,
                               imgpath = imgpath,
                               scores = scores,
                               imgpath2 = imgpath2,
                               name=name,
                               img_final=img_final
        )
    else:
        return render_template('index.html')
    
    


if __name__=="__main__":
    app.run("0.0.0.0",debug=True)
