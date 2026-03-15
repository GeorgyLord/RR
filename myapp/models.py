from django.db import models
# import hashlib
# import secrets
# from datetime import datetime, timedelta
from django.contrib.auth.models import AbstractUser
# from django.contrib.auth.models import User

# Create your models here.
 
# class Person(models.Model):
#     name = models.CharField(max_length=20)
#     email = models.CharField(max_length=30)
#     password = models.CharField(max_length=30)
#     age = models.IntegerField()


class Recipe(models.Model):
    """
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
    """
    
    Id_Recipe = models.IntegerField(null=True, blank=True)
    URL = models.CharField(max_length=500)
    Name_recipe = models.CharField(max_length=255)
    Description = models.TextField()
    Author = models.CharField(max_length=255)
    Cooking_time = models.IntegerField(default=0)
    Likes = models.IntegerField(default=0)
    Dislikes = models.IntegerField(default=0)
    Safes = models.IntegerField(default=0)
    Type_recipe = models.CharField(max_length=100)
    Tags = models.TextField()  # Для SQLite храним как текст
    Count_ingredients = models.IntegerField(default=0)
    Ingredients = models.TextField()  # Храним как текст
    Portions = models.IntegerField(default=0)
    Calorie_content = models.FloatField(default=0.0)
    Squirrels = models.FloatField(default=0.0)
    Fats = models.FloatField(default=0.0)
    Carbohydrates = models.FloatField(default=0.0)
    Steps_text = models.TextField()
    Steps_images = models.TextField()
    Url_steps_images = models.TextField()
    Images_recipe = models.TextField()
    Url_images_recipe = models.TextField()
    Number_page = models.IntegerField(default=0)



class CustomUser(AbstractUser):
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    email = models.CharField(blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    id_liked_recipes = models.JSONField(default=list, blank=True)
    created_recipes = models.ManyToManyField(
        'Recipe', 
        related_name='creators',
        blank=True,
        verbose_name='Созданные рецепты'
    )

    def __str__(self):
        return self.username
    




class RecipeReaction(models.Model):
    REACTION_CHOICES = [
        ('like', 'Нравится'),
        ('dislike', 'Не нравится'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)
    reaction = models.CharField(max_length=12, choices=REACTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'recipe']  # один пользователь - одна реакция на рецепт
