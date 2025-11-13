from django.shortcuts import render
import csv
import os
from django.conf import settings
from django.http import HttpResponse
import pandas as pd
from django.template import loader

import ast
  
  
def home(request):
    return render(request, 'home/home.html')
    # return HttpResponse("<h1>ПРИВЕТ</h1>")

def index(request):
    import test_10_best
    index_person = int(request.GET.get('user_id', 270))
    count_top = int(request.GET.get('top', 5))
    print(index_person, count_top)
    df = test_10_best.start_fun(n=index_person, top=count_top)
    # return render(request, 'a.html')
    return HttpResponse(df.to_html())

def card(request, id):
    template = loader.get_template('b.html')
    df = pd.read_csv('dataset/data.csv')
    df_id = df[df['id']==id].iloc[0]
    context = df_id.to_dict()
    if context["Images_recipe"] != '[]':
        context['Images_recipe'] = ast.literal_eval(context['Images_recipe'])[0][1]
    else:
        context['Images_recipe']
    # print('[!!]', ast.literal_eval(context['Images_recipe'])[0][1])
    # context = {
    #     'Name_recipe': df_id['Name_recipe'].iloc[0],
    #     'Description': df_id['Description'].iloc[0],
    #     "Author": df_id['Author'].iloc[0],
    #     "Cooking_time": df_id['Cooking_time'].iloc[0],
    #     "Likes": df_id['Likes'].iloc[0],
    #     "Dislikes": df_id['Dislikes'].iloc[0],
    #     "Safes": df_id['Safes'].iloc[0],
    #     'Type_recipe': df_id['Type_recipe'].iloc[0],
    #     'Tags': df_id['Tags'].iloc[0],
    #     'Count_ingredients': df_id['Count_ingredients'].iloc[0],
    #     'Ingredients': df_id['Ingredients'].iloc[0],
    #     'Pontions': df_id['Pontions'].iloc[0],
    #     'Calorie_content': df_id['Calorie_content'].iloc[0],
    #     'Squirrels': df_id['Squirrels'].iloc[0],
    #     'Fats': df_id['Fats'].iloc[0],
    #     'Carbohydrates': df_id['Carbohydrates'].iloc[0],
    #            }
    return HttpResponse(template.render(context, request))
    # return HttpResponse(f"<h1>Имя: {name}</h1>")

# def card(request):
#     return render(request, 'b.html')
#     template = loader.get_template('myapp/templates/b.html')
#     context = {'message': 'Привет, мир!'}
#     return HttpResponse(template.render(context, request))