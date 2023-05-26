from django.shortcuts import render,redirect
from django.http import HttpResponse
from service_1.models import Post,Category
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required

import pandas as pd

# Create your views here.
def home(request):
    # return HttpResponse("hi")

    #load all posts from DB
    posts = Post.objects.all()[:11] # first 10 posts
    # print(posts)

    cats = Category.objects.all()
    data={
        'posts':posts,
        'cats':cats
    }
    return render(request,'home.html',data)

def index(request):
    return render(request,'index.html')

def aboutPage(request):
    return render(request,'about.html')

def post(request,url):
    # fetch data from view
    post = Post.objects.get(url=url)
    # print(post)
    return render(request, 'posts.html',{'post':post})

@login_required(login_url='login') # Cannot bruteforce to dashboard page, will default to login page
def dashboard(request):
    return render(request, 'dashboard.html')

def signupPage(request):
    if request.method == 'POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        password=request.POST.get('password')
        confirm_password=request.POST.get('confirm-password')

        # print(uname,email,password,confirm_password)

        if(password!=confirm_password):
            return HttpResponse("Passwords Do Not Match!")
        else:
            my_user = User.objects.create_user(uname,email,password)
            my_user.save()

            # return HttpResponse("User has been created successfully!")
            return redirect('login')

    return render(request, 'signup.html')

def loginPage(request):
    if request.method == 'POST':
        uname=request.POST.get('username')
        password=request.POST.get('password')

        # print(uname,password)

        user=authenticate(request, username=uname, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            context = {'error': 'Invalid username or password'}
            return render(request, 'login.html', context=context)
    return render(request, 'login.html')

def logoutPage(request):
    logout(request)
    return redirect('login')

# @login_required(login_url='login')
def services(request):
    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)[0]
    sp500_symbols = data_table["Symbol"].to_list()
    print(type(sp500_symbols))
    return render(request,'services_home.html', {'sp500_symbols': sp500_symbols})