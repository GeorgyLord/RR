import os
import sys
import django

# Добавляем родительские директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # до проекта
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # до приложения

# Настраиваем Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()
from django.contrib.auth.models import User
# Все суперпользователи
superusers = User.objects.filter(is_superuser=True)
print(f"Всего суперпользователей: {superusers.count()}")

# Подробная информация
for user in superusers:
    print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}, Дата создания: {user.date_joined}")