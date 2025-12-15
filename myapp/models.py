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
    Pontions = models.IntegerField(default=0)
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



# from django.contrib.auth.models import AbstractUser

# class CustomUser(AbstractUser):
#     phone = models.CharField(max_length=20, blank=True)
#     # avatar = models.ImageField(upload_to='avatars/', blank=True)
    
#     class Meta:
#         # Это опционально, но помогает
#         db_table = 'custom_user'  # Имя таблицы в БД
    
#     def __str__(self):
#         return self.email
    

# class MyUser(models.Model):
#     class Meta:
#         verbose_name = 'Пользователь'
#         verbose_name_plural = 'Пользователи'
#     username = models.CharField(max_length=50, unique=True)
#     email = models.EmailField(unique=True)
#     password_hash = models.CharField(max_length=128)  # Для хэша пароля
#     is_active = models.BooleanField(default=True)
#     # created_at = models.DateTimeField(auto_now_add=True)
    
#     # Для "запомнить меня"
#     session_token = models.CharField(max_length=64, blank=True)
#     session_expires = models.DateTimeField(null=True)
    
#     def set_password(self, raw_password):
#         """Хэширование пароля"""
#         salt = secrets.token_hex(16)
#         # Простой хэш (в реальности используйте bcrypt/argon2)
#         hash_obj = hashlib.sha256(f"{salt}{raw_password}".encode())
#         self.password_hash = f"{salt}${hash_obj.hexdigest()}"
    
#     def check_password(self, raw_password):
#         """Проверка пароля"""
#         if '$' not in self.password_hash:
#             return False
#         salt, stored_hash = self.password_hash.split('$')
#         test_hash = hashlib.sha256(f"{salt}{raw_password}".encode()).hexdigest()
#         return test_hash == stored_hash
    
#     def create_session(self, remember=False):
#         """Создание сессии"""
#         self.session_token = secrets.token_hex(32)
#         if remember:
#             self.session_expires = datetime.now() + timedelta(days=30)
#         else:
#             self.session_expires = datetime.now() + timedelta(hours=2)
#         self.save()
#         return self.session_token
    
#     def validate_session(self, token):
#         """Проверка сессии"""
#         return (self.session_token == token and 
#                 self.session_expires > datetime.now())
    
#     def logout(self):
#         """Выход"""
#         self.session_token = ''
#         self.session_expires = None
#         self.save()
    
#     def __str__(self):
#         return self.username


class CustomUser(AbstractUser):
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # Добавляем дополнительные поля
    # username = None
    # email = None
    email = models.CharField(blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    id_liked_recipes = models.JSONField(default=list, blank=True)

    def __str__(self):
        return self.username
    

# class Interactions(models.Model):
#     class Meta:
#         db_table = 'interactions'
    
#     # id_user = models.IntegerField()
#     # id_user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
#     id_recipe = models.IntegerField()
#     interaction = models.CharField()
    
#     def __str__(self):
#         # return self.q.username
#         return self.id_recipe



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
