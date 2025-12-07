from myapp.models import Recipe
import pandas as pd

print(1)
r = Recipe.objects.create(
    Id = 0,
    URL = "",
    Name_recipe = "",
    Description = "",
    Author = "",
    Cooking_time = 0,
    Likes = 0,
    Dislikes = 0,
    Safes = 0,
    Type_recipe = "",
    Tags = "",
    Count_ingredients = 0,
    Ingredients = 0,
    Pontions = 0,
    Calorie_content = 0,
    Squirrels = 0,
    Fats = 0,
    Carbohydrates = 0,
    Steps_text = "",
    Steps_images = "",
    Url_steps_images = "",
    Images_recipe = "",
    Url_images_recipe = "",
    Number_page = 0,
)