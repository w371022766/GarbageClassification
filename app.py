# coding=utf-8
from flask import Flask, request, jsonify
from helper import object_detection,object_translate
import flask_cors
import base64
import time
import numpy as np
import cv2
import os
from gensim.models.keyedvectors import KeyedVectors

app = Flask(__name__)
flask_cors.CORS(app, supports_credentials=True)

root_dir = os.path.abspath(os.path.dirname(__file__))
image_dir = os.path.join(root_dir, 'static/image/')

recoverableItems = ['报纸', '纸箱', '书本', '纸袋', '信封', '塑料瓶', '玩具', '油桶', '乳液罐', '食品保鲜盒', '衣架', '泡沫塑料', '酒瓶', '玻璃杯', '玻璃放大镜', '窗玻璃', '碎玻璃', '易拉罐', '锅', '螺丝刀', '刀', '刀片', '指甲钳', '皮鞋', '衣服', '床单', '枕头', '包', '毛绒玩具', '电路板', '电线', '插座', '积木', '砧板']
hazardousItems = ['充电电池', '蓄电池', '纽扣电池', '碱性电池', '铂酸电池', '荧光灯', '节能灯', '卤素灯', '过期药物', '药品包装', '药片', '过期胶囊药品', '废油漆桶', '染发剂壳', '过期指甲油', '洗甲水', '水银血压计', '水银体温计', '消毒剂', '老鼠药', '杀虫喷雾', 'X光片', '感光胶片', '胶片底片']
householdItems = ['剩饭剩菜', '蛋糕饼干', '面包', '动物内脏', '苹果核', '鸡肉', '鱼', '虾', '鸡蛋', '蛋壳', '干果仁', '蔬菜', '大米', '花卉', '宠物饲料', '中药药渣', '水果']
residualItems = ['餐巾纸', '烟蒂', '卫生间用纸', '陶瓷花盆', '胶带', '橡皮泥', '创可贴', '笔', '灰尘', '眼镜', '头发', '内衣裤', '防碎气泡罐', '污损纸张', '旧毛巾', '敲碎陶瓷碗', '无损塑料袋']

garbagedict = {}

for item in recoverableItems:
    garbagedict[item] = "可回收垃圾"
for item in hazardousItems:
    garbagedict[item] = "有害垃圾"
for item in householdItems:
    garbagedict[item] = "湿垃圾"
for item in residualItems:
    garbagedict[item] = "干垃圾"

#print(len(garbagedict))

#load word embedding model
file = r"45000-small.txt"
model = KeyedVectors.load_word2vec_format(file, binary=False)

def object_closest(object_chinese):
    bestres = 0
    word_closest = ''
    cnt = 0
    for (key, value) in garbagedict.items():
        try:
            similarity = model.similarity(object_chinese, key)
            if similarity > bestres:
                bestres = similarity
                word_closest = key
        except Exception as e:
            #print('{0} and {1} cannot calculate similarity because one not in vocabulary'.format(object_chinese, key))
            cnt += 1
    #print('{0} times cannot calculate similarity'.format(cnt))
    print('best word: {0}, similarity: {1}'.format(word_closest, bestres))
    if bestres > 0.4:
        return word_closest
    else:
        return ''


@app.route('/post/api', methods=['GET','POST'])
def post_api():
    if request.method == 'POST':
        # 获取数据
        data = request.form
        # print(data)
        image = data['image']
        image = base64.b64decode(image)
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # 存图片
        #file_name_by_time = str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime(time.time())))
        #file_path = os.path.join(image_dir, '{}.jpg'.format(file_name_by_time))
        file_path = os.path.join(image_dir, 'temp.jpg')
        cv2.imwrite(file_path, image)

        with open(file_path, 'rb') as f:
            image_bin = f.read()
            detection_result = object_detection(image_bin)
        for obj_json in detection_result['objects']:
            obj_english = obj_json['object']
            obj_json['object_english'] = obj_english
            obj_chinese = object_translate(obj_english)
            obj_closest = object_closest(obj_chinese)
            obj_json['object'] = obj_chinese
            obj_json['object_closest'] = obj_closest
            if obj_closest and (not obj_closest.isspace()):
                obj_json['classification'] = garbagedict[obj_closest]
            else:
                obj_json['classification'] = ''
        reply = {
            'detection_result': detection_result
        }
        print(obj_json)
        return jsonify(reply)
    else:
        return "You need to use post method"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443, ssl_context='adhoc')
    #app.run(host='0.0.0.0', port=443, ssl_context=('certificate.crt', 'server.key'))
