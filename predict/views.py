from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

from predict.needs.processData import *
from predict.needs.predict import *
from predict.needs.recommendation import *
from predict.needs.jsonProcess import *
from predict.needs.exception import * 
from BurpyIC.settings import CATEGORY_LIST
import json

def inappropriate_access(request):
    return render(request, 'inappropriate.html', {})

def image_classification(request):
    # try-except의 주석
    try:
        img_file = decode_json(request, 'image')
        result = inception_predict(img_file)
        result = encode_json(result)
        print(result)
    
    except Exception as e:
        except_info = config_except_info()
        print_except_info(except_info)
        result = encode_json(except_info)
    
    return JsonResponse(result, safe=False)

def on_recommend_train_data(request):
    result = save_train_data(request.body)
    result = encode_json(result)
    return JsonResponse(result, safe=False)

def on_recommend_train(request):
    user_list = json.loads(request.body.decode("utf-8"))
    for user in user_list:
        for category in CATEGORY_LIST: # FIX: train 가능한 category만 순회
            result = train_recommendation(user['_id'], category)
    
    result = encode_json(result)
    return JsonResponse(result, safe=False)

def on_recommend_predict_data(request):
    result = save_predict_data(request.body)
    result = encode_json(result)
    return JsonResponse(result, safe=False)

def on_recommend_predict(request):
    user_list = json.loads(request.body.decode("utf-8"))
    for user in user_list:
        for category in CATEGORY_LIST: # FIX: predict 가능한 category만 순회
            result = predict_recommendation(user['_id'], category)
        
    result = encode_json(result)
    return JsonResponse(result, safe=False)

def on_recommend_predict_result(request):
    user_id = json.loads(request.body.decode("utf-8"))
    result = fetch_predict_result(user_id)
    result = encode_json(result)
    return JsonResponse(result, safe=False)