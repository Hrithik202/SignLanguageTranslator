from django.shortcuts import render

def home(request):
    output=None # this gives default value even if form is not submitted
    if request.method == "POST":
        user_input= request.POST.get ("user_input")
        output= user_input
    return render(request, 'translator/home.html', {"output":output})


def contact(request):
    return render(request, 'translator/contact.html')

def about(request):
    return render(request, 'translator/about.html')