from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render,redirect
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
def result(request):
	file=open('model.pkl','rb')
	model = pickle.load(file)
	file.close()
	file=open("vectorizer.pkl",'rb')
	vectorizer=pickle.load(file)
	file.close()
	s=request.POST['subject']
	testdata=vectorizer.transform([s])
	out=model.predict(testdata)
	return render(request,"result.html",{'data':out[0]})

def home(request):
	return render(request,"home.html")