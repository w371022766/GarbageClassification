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

recoverableItems = ['厨房电器', '电子产品', '家电', '报纸', '纸箱', '书本', '纸袋', '信封', '塑料瓶', '玩具', '油桶', '乳液罐', '食品保鲜盒', '衣架', '泡沫塑料', '酒瓶', '玻璃杯', '玻璃放大镜', '窗玻璃', '碎玻璃', '易拉罐', '锅', '螺丝刀', '刀', '刀片', '指甲钳', '皮鞋', '衣服', '床单', '枕头', '包', '毛绒玩具', '电路板', '电线', '插座', '积木', '砧板']
hazardousItems = ['充电电池', '蓄电池', '纽扣电池', '碱性电池', '铂酸电池', '荧光灯', '节能灯', '卤素灯', '过期药物', '药品包装', '药片', '过期胶囊药品', '废油漆桶', '染发剂壳', '过期指甲油', '洗甲水', '水银血压计', '水银体温计', '消毒剂', '老鼠药', '杀虫喷雾', 'X光片', '感光胶片', '胶片底片']
householdItems = ['剩饭剩菜', '蛋糕饼干', '面包', '动物内脏', '苹果核', '鸡肉', '鱼', '虾', '鸡蛋', '蛋壳', '干果仁', '蔬菜', '大米', '花卉', '宠物饲料', '中药药渣', '水果']
residualItems = ['餐巾纸', '烟蒂', '卫生间用纸', '陶瓷花盆', '胶带', '橡皮泥', '创可贴', '笔', '灰尘', '眼镜', '头发', '内衣裤', '防碎气泡罐', '污损纸张', '旧毛巾', '敲碎陶瓷碗', '无损塑料袋']

recoverableItems_1 = ["废纸张", "纸板箱", "报纸", "废弃书本", "快递纸袋", "打印纸", "信封", "广告单", "利乐包", "废塑料", "饮料瓶", "奶瓶", "洗发水瓶", "乳液罐", "食用油桶", "塑料碗", "塑料盆", "塑料盒", "食品保鲜盒", "收纳盒", "塑料玩具", "塑料积木", "塑料模型", "塑料衣架", "施工安全帽", "PE塑料", "pvc", "亚克力板", " 塑料卡片", "密胺餐具", "kt板", "泡沫塑料", "水果网套", "废玻璃制品", "调料瓶", "酒瓶", "化妆品瓶", "玻璃杯", "窗玻璃", "玻璃制品", "放大镜", "玻璃摆件", "碎玻璃", "废金属", "金属瓶罐", "易拉罐", "食品罐", "食品桶", "金属厨具", "菜刀", "锅", "金属工具", "刀片", "指甲剪", "螺丝刀", "金属制品", "铁钉", "铁皮", "铝箔", "废织物", "旧衣服", "床单", "枕头", "棉被", "皮鞋", "毛绒玩具", "布偶", "棉袄", "包", "皮带", "丝绸制品", "电路板", "主板", "内存条", "充电宝", "电线", "插头", "木制品", "积木", "砧板", "砧板", "手机", "面霜瓶"]
hazardousItems_1 = ["充电电池", "镉镍电池", "铅酸电池", "蓄电池", "纽扣电池", "荧光（日光）灯管", "卤素灯", "过期药物", "药物胶囊", "药片", "药品内包装", "废油漆桶", "染发剂壳", "过期的指甲油", "洗甲水", "废矿物油及其包装物", "废含汞温度计", "废含汞血压计", "水银血压计", "水银体温计", "水银温度计", "废杀虫剂及其包装", "老鼠药", "毒鼠强", "杀虫喷雾罐", "废胶片及废相纸", "x光片等感光胶片", "相片底片"]
householdItems_1 = ["食材废料", "谷物及其加工食品", "米", "米饭", "面", "面包", "豆类", "肉蛋及其加工食品", "鸡", "鸭", "猪", "牛", "羊肉", "蛋", "动 物内脏", "腊肉", "午餐肉", "蛋壳", "水产及其加工食品", "鱼", "鱼鳞", "虾", "虾壳", "鱿鱼", "蔬菜", "绿叶菜", "根茎蔬菜", "菌 菇", "调料", "酱料", "剩菜剩饭", "火锅汤底", "沥干后的固体废弃物", "鱼骨", "碎骨", "碎骨头", "茶叶渣", "咖啡渣", "过期食品", "糕饼", "糖果", "风干食品", "肉干", "红枣", "中药材", "粉末类食品", "冲泡饮料", "面粉", "宠物饲料", "瓜皮果核", "水果果肉", "椰子肉", "水果果皮", "西瓜皮", "桔子皮", "苹果皮", "水果茎枝", "葡萄枝", "果实", "西瓜籽", "花卉植物", "家养绿植", "花卉", "花瓣", "枝叶", "中药药渣", "奶茶中的珍珠", "水果", "麻辣烫", "小龙虾", "粽子馅", "猫粮", "鸡蛋壳", "蛋壳", "瓜子", "瓜子壳"]
residualItems_1 = ["餐巾纸", "卫生间用纸", "尿不湿", "猫砂", "狗尿垫", "污损纸张", "烟蒂", "干燥剂", "污损塑料", "尼龙制品", "编织袋", "防碎气泡膜", "大骨头", "硬贝壳", "硬果壳", "椰子壳", "榴莲壳", "核桃壳", "玉米衣", "甘蔗皮", "硬果实", "榴莲核", "菠萝蜜核", "毛发", "灰土", "炉渣", "橡皮泥", "太空沙", "带胶制品", "胶水", "胶带", "花盆", "毛巾", "一次性餐具", "镜子", "陶瓷制品", "竹制品", "竹篮", "竹筷", "牙签", "成分复杂的制品", "伞", "笔", "眼镜", "打火机", "奶茶杯", "奶茶塑料盖", "婴儿尿布", "一次性尿布", "面膜", "吸管", "粽子皮", "粽子壳", "粽子叶", "粽子包扎线", "干冰", "干冰袋", "外卖盒", "塑料袋"]


garbagedict = {}

for item in recoverableItems:
    garbagedict[item] = "可回收垃圾"
for item in hazardousItems:
    garbagedict[item] = "有害垃圾"
for item in householdItems:
    garbagedict[item] = "湿垃圾"
for item in residualItems:
    garbagedict[item] = "干垃圾"

for item in recoverableItems_1:
    garbagedict[item] = "可回收垃圾"
for item in hazardousItems_1:
    garbagedict[item] = "有害垃圾"
for item in householdItems_1:
    garbagedict[item] = "湿垃圾"
for item in residualItems_1:
    garbagedict[item] = "干垃圾"    

print(len(garbagedict))

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
        file_name_by_time = str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime(time.time())))
        file_path = os.path.join(image_dir, '{}.jpg'.format(file_name_by_time))
        #file_path = os.path.join(image_dir, 'temp.jpg')
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
        print(detection_result)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print('source image file does not exists')
        return jsonify(reply)
    else:
        return "You need to use post method"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443, ssl_context='adhoc')
    #app.run(host='0.0.0.0', port=443, ssl_context=('certificate.crt', 'server.key'))
