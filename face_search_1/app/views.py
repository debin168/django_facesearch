from django.shortcuts import render
from django import forms
# Create your views here.
from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from app.models import Image
import sys
import urllib
import uuid
import app.src.search as  search

class ImageForm(forms.Form):
    Img = forms.FileField()
# Create your views here.
def register(request):
    if request.method =="POST":
        form=ImageForm(request.POST,request.FILES)
        if form.is_valid():
             img = form.cleaned_data['Img']
             imgObj =Image()
             imgObj.uid=uuid.uuid1()
             imgObj.path=img

             imgObj.save()
             return  HttpResponse('ok')
    else:
        form=ImageForm()
    return render_to_response('index.html',{'form':form})


def get(request):
    likeImgPath = []
    path=''
    if request.method=="POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
             img = form.cleaned_data['Img']
             uid=uuid.uuid1()
             imgObj =Image()
             imgObj.uid=uid
             imgObj.path=img
             imgObj.save()
             obj=Image.objects.get(uid=uid)
             path= obj.path.url.encode('utf-8')

             sys.argv = ['compare.py', '/home/ubuntu/face_search_1/face_search_1/app/src/20170512-110547.pb',
                         '/home/ubuntu/face_search_1/face_search_1/'+path]

             args = search.parse_arguments(sys.argv[1:])
             likesImg = search.main(args)

             for i in range(10):
                  filepath = likesImg[i, 0]
                  likeImgPath.append(filepath)
    else:
         form = ImageForm()
    return render_to_response('get.html', {'form': form,'likesImg':likeImgPath,'path':path})



