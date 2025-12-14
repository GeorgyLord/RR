## Активация окружения
`.venv\Scripts\activate` - CMD

`source .venv/Scripts/activate` - BASH

## Создание Django-проекта
`django-admin startproject myproject`

## Запуск Django-проекта
`python manage.py runserver`

`python manage.py makemigrations`

`python manage.py migrate`


## Для формирования requirements.txt
`pip freeze > requirements.txt`

## Скачивание необходимых библиотек
`pip install -r requirements.txt`

## У становленные пакеты
`pip list`

## Посмотреть версию Python
`python --version`


## Создание суперпользователя
`python manage.py createsuperuser`


## Удаление базы данных
`rm db.sqlite3`

## Удаление всх миграции
```
find . -name "*.pyc" -delete
find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
```

## Переустанова Django
`pip install --force-reinstall django`

## Проверка проекта на ошибки
`python manage.py check`