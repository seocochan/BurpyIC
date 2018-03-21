from django.shortcuts import render
import predict.predict as pd
from remindDjango.settings import MEDIA_ROOT

# Create your views here.
def maindoor(request):
    return render(request, 'main.html', {})

def result(request):
    im = request.FILES.getlist("file")
    dir = MEDIA_ROOT + '/'+ im[0].name
    
    with open(dir, 'wb+') as dest:
        for chunk in im[0].chunks():
            dest.write(chunk)

    data = pd.imageProcess(dir)

    result = pd.CNNprediction(data)
    
    print(result, "입니다!")
    return render(request, 'result.html', {'result':result})
