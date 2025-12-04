from django.shortcuts import render
import csv
import os
from django.conf import settings
from django.http import HttpResponse
import pandas as pd
from django.template import loader
from test_10_best import start_fun
import ast
import json
  
cur_per_id = 71
tdf = start_fun(cur_per_id, 500)
with open("images_to_local.json", "r") as f:
    file_data_images_to_local = json.load(f)


def home(request):
    tls = tdf['id'].tolist()
    context = {}
    df_data = pd.read_csv('dataset/data.csv')
    list_link_for_images_recipes = []
    for i in range(len(tls)):
        temp_list = [tls[i]]
        if df_data[df_data['id']==tls[i]].iloc[0]['Images_recipe'] != '[]':
            df_id = ast.literal_eval(df_data[df_data['id']==tls[i]].iloc[0]['Images_recipe'])[0][1]
            # print(f'images/images/{file_data_images_to_local.get(df_id)}')
            if file_data_images_to_local.get(df_id) == None:
                temp_list.append(None)
            else:
                temp_list.append(f'images/images/{file_data_images_to_local.get(df_id)}')
        df_name_recipe = df_data[df_data['id']==tls[i]].iloc[0]['Name_recipe'] # Name_recipe
        temp_list.append(df_name_recipe)
        list_link_for_images_recipes.append(temp_list)
    
    context['card_recipe']=list_link_for_images_recipes
    template = loader.get_template('home/home.html')
    return HttpResponse(template.render(context, request))
    # return render(request, 'home/home.html')
    # return HttpResponse("<h1>ПРИВЕТ</h1>")

def settings(request):
    template = loader.get_template('settings/settings.html')
    df_data = pd.read_csv('dataset/data.csv')
    df_interaction = pd.read_csv('dataset/interaction.csv')
    # print(df_interaction[df_interaction['user_id']==cur_per_id])
    intranction_current_person = df_interaction[df_interaction['user_id']==cur_per_id]
    list_neg = intranction_current_person[intranction_current_person['interaction'] == -1]['item_id'].tolist()
    list_pos = intranction_current_person[intranction_current_person['interaction'] == 1]['item_id'].tolist()
    # df_interaction_current_person = list(df_interaction[df_interaction['user_id']==71]['item_id'])
    # print(df_interaction_current_person)
    print(intranction_current_person)
    print(list_neg, list_pos)
    id_list_likes_recipes = {
        "list_neg":list_neg,
        "list_pos":list_pos,
        }
    context = id_list_likes_recipes
    return HttpResponse(template.render(context, request))
    # return render(request, 'settings/settings.html')

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
        # temp_image_recipe = context['Images_recipe']
        # print(df['id'])
        local_link = file_data_images_to_local.get(context['Images_recipe'])
        # print(local_link)
        if local_link != None:
            context['Images_recipe'] = 'images/images/'+local_link
        else:
            context['Images_recipe'] = 'images/not_image/not_image_recipe.png'
    else:
        context['Images_recipe'] = 'images/not_image/not_image_recipe.png'
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