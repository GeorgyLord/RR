from django.db import models

# Create your models here.
 
class Person(models.Model):
    name = models.CharField(max_length=20)
    email = models.CharField(max_length=30)
    password = models.CharField(max_length=30)
    age = models.IntegerField()

class Recipe(models.Model):
    Id_Recipe = models.IntegerField(null=True)
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