# myapp/code/your_script.py
import os
import sys
import django

# Добавляем родительские директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # до проекта
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # до приложения

# Настраиваем Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

# Теперь работаем с БД
from myapp.models import Recipe
# records = Recipe.objects.all()
# print(records)

r = Recipe.objects.create(
    Id_Recipe = 0,
    URL = "empty",
    Name_recipe = "empty",
    Description = "empty",
    Author = "empty",
    Cooking_time = 0,
    Likes = 0,
    Dislikes = 0,
    Safes = 0,
    Type_recipe = "empty",
    Tags = "empty",
    Count_ingredients = 0,
    Ingredients = 0,
    Pontions = 0,
    Calorie_content = 0,
    Squirrels = 0,
    Fats = 0,
    Carbohydrates = 0,
    Steps_text = "empty",
    Steps_images = "empty",
    Url_steps_images = "empty",
    Images_recipe = "empty",
    Url_images_recipe = "empty",
    Number_page = 0,
)