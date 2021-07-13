from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pickle
from models.predict import predict_on_cracks
from django.http import FileResponse
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth



global fp1

fp1=""
# Create your views here.
def index(request):    
    context={"a":1}
    return render(request, 'index.html',context)

def intro(request):    
    context={"a":1}
    return render(request, 'intro.html',context)


def output(request):
    
    fileobj = (request.FILES['filepath'])
    fs = FileSystemStorage()

    filepathname = fs.save(fileobj.name,fileobj)
    filepathname = fs.url(filepathname)
    testimage='.'+filepathname
    
    fp,outimg,pcrack,s=predict_on_cracks(testimage)
    global fp1
    fp1 = fp
    pcrack = pcrack*100
    #print("nehallllllllllll : ",filepathname)
    pcrack = np.round(pcrack,3)
    context={"filepathname":filepathname, "percentagecrack":pcrack,"string":s}
    return render(request, 'output.html',context)

def predicted(request):    
    context={}

    return render(request, 'predicted.html',context)

def sendfile(response): 
    global fp1
    
    img = open(fp1, 'rb')

    response = FileResponse(img)

    return response

def login(request):
    if request.method == 'POST':
        password = request.POST['password']
        username= request.POST['username']
        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('/index')
        else:
            messages.info(request,"invalid credentials!")    
            return redirect('login')
    else:    
        return render(request, 'login.html')



def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']
        username= request.POST['username']
        
        if password==password2:
            if User.objects.filter(username=username).exists():
                print("username taken")
                messages.info(request,"username taken!")
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                print("email alredy taken")
                messages.info(request,"email already taken!")
                return redirect('register')
            else:
                user= User.objects.create_user(username=username, password=password,email=email,first_name=first_name,last_name=last_name)
                user.save()
                print('user created')
                messages.info(request,"user created")
                return redirect('login')
        else:
            print("passwords do not macth!!")
            return redirect('register')
        return redirect('/')

    else:
        return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('index')
