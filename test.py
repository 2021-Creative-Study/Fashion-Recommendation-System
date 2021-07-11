from feature_extractor import FeatureExtractor
import json
import os

fe = FeatureExtractor()
# li는 이미지 경로 
# final 은 입력받은 라벨링에 대한 이미지 경로 

def result(input_label):
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
        if query==input_label:
            cnt+=1
            if cnt == 3:
                break
            img_final.append(os.path.join(img_path, img_li[i]))

    return img_final 
    

print(result())
print('끝')

