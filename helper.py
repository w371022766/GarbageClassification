import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import requests
import uuid
import configparser

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
        ret = translation_dict['text']
    return ret