from django.db import models

# Create your models here.


class Recipe(models.Model):
    Id = models.IntegerField()
    URL = models.CharField()
    Name_recipe = models.CharField()
    Description = models.CharField()
    Author = models.CharField()
    Cooking_time = models.IntegerField()
    Likes = models.IntegerField()
    Dislikes = models.IntegerField()
    Safes = models.IntegerField()
    Type_recipe = models.CharField()
    Tags = models.CharField()
    Count_ingredients = models.IntegerField()
    Ingredients = models.IntegerField()
    Pontions = models.IntegerField()
    Calorie_content = models.IntegerField()
    Squirrels = models.IntegerField()
    Fats = models.IntegerField()
    Carbohydrates = models.IntegerField()
    Steps_text = models.CharField()
    Steps_images = models.CharField()
    Url_steps_images = models.CharField()
    Images_recipe = models.CharField()
    Url_images_recipe = models.CharField()
    Number_page = models.IntegerField()