from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

from predict.needs.processData import *
from predict.needs.predict import *
from predict.needs.jsonProcess import *
from predict.needs.exception import * 


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















"""
    im = request.FILES.get('file')
    name = im.name

    # Use the following code if you want to store in the MEDIA path.
    
    print(type(im))
    dir = MEDIA_ROOT + '/'+ name
    
    with open(dir, 'wb+') as dest:
        for chunk in im.chunks():
            dest.write(chunk)
    
    # Process and save the image.
    image = Image.open(im)
    image = pd.imageProcess(image)
    saveimg(image, name)

    # Image datalization.
    data = pd.datalization(image)
    result = pd.CNNprediction(data)

    print("Maybe... ", result)
    return render(request, 'result.html', {'result':result, 'image':name})

def saveimg(image, name):
    # Save the image as a temporary measure.
    path = os.path.join(MEDIA_ROOT, 'image')
    image.save(os.path.join(path, name))
"""