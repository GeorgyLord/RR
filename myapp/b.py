# import sys
# import os
# # Добавляем родительскую директорию в sys.path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# # Или для родителя родительской директории:
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import django
import sys

# # Добавляем путь к проекту
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# # Настройка Django
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
# django.setup()

import pandas as pd
from pathlib import Path
# from .. import models
# from models import Recipe


current_path = Path.cwd()
BASE_DIR = Path(__file__).resolve().parent.parent
# print(BASE_DIR)
sys.path.append(f'{BASE_DIR}')
os.environ['DJANGO_SETTINGS_MODULE'] = 'myproject.settings'
django.setup()

from myapp.models import Recipe

df = pd.read_csv(BASE_DIR / 'dataset/data.csv')
# print(dfq['Name_recipe'])

# for i in range(4004, len(df)):
if 1:
    i = 4003
    try:
        dfq = df.iloc[i]
        recipe = Recipe.objects.create(
            Id_Recipe = dfq['id'],
            URL = dfq['URL'],
            Name_recipe = dfq['Name_recipe'],
            Description = dfq['Description'],
            Author = dfq['Author'],
            Cooking_time = dfq['Cooking_time'],
            Likes = dfq['Likes'],
            Dislikes = dfq['Dislikes'],
            Safes = dfq['Safes'],
            Type_recipe = dfq['Type_recipe'],
            Tags = dfq['Tags'],
            Count_ingredients = dfq['Count_ingredients'],
            Ingredients = dfq['Ingredients'],
            Pontions = dfq['Pontions'],
            Calorie_content = dfq['Calorie_content'],
            Squirrels = dfq['Squirrels'],
            Fats = dfq['Fats'],
            Carbohydrates = dfq['Carbohydrates'],
            Steps_text = dfq['Steps_text'],
            Steps_images = dfq['Steps_images'],
            Url_steps_images = dfq['Url_steps_images'],
            Images_recipe = dfq['Images_recipe'],
            Url_images_recipe = dfq['Url_images_recipe'],
            Number_page = dfq['Number_page'],
        )
    except:
        print(i)