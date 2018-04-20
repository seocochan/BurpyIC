from django.http import HttpResponse
import json
import base64

# decode Json type data
def decode_json(request, json_field):
    json_data = json.loads(request.body.decode('utf-8'))
    json_data = json_data[json_field]
    return base64.b64decode(json_data)

# encode Json
def encode_json(target):
    if len(target) != 0:
        result = target
    else:
        e = Exception('dict\'s length is 0')
        raise e
    result_string = json.dumps(result)
    return result_string