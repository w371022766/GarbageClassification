import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import requests
import uuid
import configparser
import threading

config = configparser.ConfigParser()
config.read('config.ini')

def object_detection(image_bin):
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',  # application/octet-stream, multipart/form-data or application/json
        'Ocp-Apim-Subscription-Key': config['Subscription-Key']['computevision'],
    }
    conn = http.client.HTTPSConnection('eastasia.api.cognitive.microsoft.com')
    params = urllib.parse.urlencode({
    })
    try:
        conn.request("POST", "/vision/v2.0/detect?%s" % params, image_bin, headers)
        response = conn.getresponse()
        ret = json.loads(response.read().decode('utf-8'))
        #print('Object Detection Result:', ret)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        ret = {"objects": "null"}
    return ret


def object_translate(obj_english):
    subscriptionKey = config['Subscription-Key']['translate']
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&to=zh'
    constructed_url = base_url + path + params
    headers = {
        'Ocp-Apim-Subscription-Key': subscriptionKey,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': obj_english
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    ret = ''
    for translation_dict in response[0]['translations']:
        ret = translation_dict['text'].replace(" ","")
    return ret


def object_detection_custom_vision(file_path):
    with open(file_path, 'rb') as f:
        image_bin = f.read()
    iteration_name = 'Iteration3'
    custom_vision_url = 'https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/59a495da-0a06-4c4e-a9d0-16fc224f86f6/detect/iterations/{}/image'.format(
        iteration_name)
    headers = {
        'Content-Type': 'application/octet-stream',
        'Prediction-key': config['Subscription-Key']['customvision']
    }
    request = requests.post(custom_vision_url, headers=headers, data=image_bin)
    response_json = request.json()
    response_json['objects'] = []
    # print(response_json)
    for obj in response_json['predictions']:
        if obj['probability'] < 0.9: continue
        obj_json = dict()
        obj_json['object'] = obj['tagName']
        obj_json['confidence'] = obj['probability']
        obj_json['rectangle'] = dict()
        obj_json['rectangle']['x'] = obj['boundingBox']['left']
        obj_json['rectangle']['y'] = obj['boundingBox']['top']
        obj_json['rectangle']['h'] = obj['boundingBox']['height']
        obj_json['rectangle']['w'] = obj['boundingBox']['width']
        response_json['objects'].append(obj_json)
    response_json['predictions'] = ''
    return response_json


class MyThread(threading.Thread):

    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None
