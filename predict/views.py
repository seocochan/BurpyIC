from django.shortcuts import render

import predict.predict as pd
from remindDjango.settings import BASE_DIR, MEDIA_ROOT
from PIL import Image

import os

# Create your views here.
def maindoor(request):
    return render(request, 'main.html', {})

def result(request):
    im = request.FILES.get('file')
    name = im.name

    # Use the following code if you want to store in the MEDIA path.
    """
    print(type(im))
    dir = MEDIA_ROOT + '/'+ name
    
    with open(dir, 'wb+') as dest:
        for chunk in im.chunks():
            dest.write(chunk)
    """
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