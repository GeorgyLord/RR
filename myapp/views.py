import csv
import os
from django.conf import settings
from django.http import HttpResponse
from collections import defaultdict
import pandas as pd
from django.template import loader
# from test_10_best import start_fun
# from SystemRecomendation_3 import get_recommendations_for_user
from SystemRecomendation_5 import get_recommendations_for_user
# from SystemRecomendation_2 import start_fun
import ast
import json
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.db.models.functions import Lower
from django.contrib import messages
from .forms import RegisterForm, LoginForm
from .models import RecipeReaction, Recipe
from .models import CustomUser
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Count
from django.http import JsonResponse
from django.db.models import Case, When, Value, IntegerField
from django.template.loader import render_to_string
# from .models import MyUser


class MyUser():
    def __init__(self):
        pass

# cur_per_id = 71
# tdf = start_fun(cur_per_id, 10)
# with open("images_to_local.json", "r") as f:
#     file_data_images_to_local = json.load(f)


# Глобальная загрузка JSON-данных для пути к изображениям
try:
    # Предполагается, что images_to_local.json находится в корне проекта или в папке, доступной для Django
    with open("images_to_local.json", "r") as f:
        file_data_images_to_local = json.load(f)
    print("images_to_local.json успешно загружен.")
except FileNotFoundError:
    print("ВНИМАНИЕ: Файл 'images_to_local.json' не найден. Изображения могут не отображаться.")
    file_data_images_to_local = {}
except json.JSONDecodeError:
    print("ОШИБКА: Не удалось декодировать images_to_local.json. Проверьте формат.")
    file_data_images_to_local = {}


def extract_url_from_string(data_string):
    """
    Безопасно парсит строку вида "[['0', 'ссылка']]" и извлекает ссылку.
    """
    if not isinstance(data_string, str):
        return None

    try:
        # Шаг 1: Парсинг строки в список Python
        parsed_list = ast.literal_eval(data_string)

        # Шаг 2: Проверка и извлечение ссылки
        # Ожидаем: [['0', 'ссылка']] -> берем [0], затем [1]
        if (isinstance(parsed_list, list) and
            len(parsed_list) > 0 and
            isinstance(parsed_list[0], list) and
                len(parsed_list[0]) > 1):

            # Возвращаем второй элемент вложенного списка
            return parsed_list[0][1]
        else:
            return None

    except (ValueError, SyntaxError, TypeError):
        # Ошибка, если строка не является корректным списком
        return None

def extract_data_from_string_2(data_string, single_item=True):
    """
    Безопасно парсит строку вида "[['0', 'данные1'], ['1', 'данные2'], ...]" 
    и извлекает данные.
    
    Args:
        data_string (str): Строка для парсинга
        single_item (bool): Если True - возвращает первый элемент (для обратной совместимости).
                           Если False - возвращает список всех данных.
    
    Returns:
        str/None или list/None: В зависимости от параметра single_item
    """
    if not isinstance(data_string, str) or not data_string.strip():
        return None if single_item else []

    try:
        # Шаг 1: Парсинг строки в список Python
        parsed_list = ast.literal_eval(data_string)
        
        # Проверяем что получили список списков
        if not isinstance(parsed_list, list):
            return None if single_item else []
        
        # Собираем все элементы
        result = []
        for item in parsed_list:
            if (isinstance(item, list) and 
                len(item) > 1 and 
                item[1] is not None):  # Проверяем наличие данных
                
                # Можем сохранить с индексом или без
                if single_item:
                    # Для обратной совместимости возвращаем первый элемент
                    return item[1]
                else:
                    result.append(item[1])
        
        # Если не нашли данных или single_item=True и нет первого элемента
        if single_item:
            return None
        return result

    except (ValueError, SyntaxError, TypeError) as e:
        # Для отладки можно раскомментировать:
        # print(f"Ошибка парсинга строки: {e}")
        return None if single_item else []


@login_required
def settings_page(request):
    """
    Отображает страницу настроек с данными пользователя и его реакциями из БД.
    """
    user = request.user
    print(user.id)
    # 1. Получение всех реакций пользователя
    # Используем select_related('recipe'), чтобы избежать N+1 запросов при доступе к Name_recipe
    user_reactions = RecipeReaction.objects.filter(
        user=user).select_related('recipe')

    # for i in list(user_reactions):
    #     print(i.id, i.reaction, i.recipe_id, i.user_id)

    # Фильтрация и получение объектов рецептов
    # liked_recipes = [
    #     reaction.recipe for reaction in user_reactions if reaction.reaction == 'like'
    # ]
    # disliked_recipes = [
    #     reaction.recipe for reaction in user_reactions if reaction.reaction == 'dislike'
    # ]
    id_liked_recipes = []
    for i in range(len(user_reactions)):
        if user_reactions[i].reaction == 'like':
            id_liked_recipes.append(user_reactions[i].recipe_id)
    liked_recipes = []
    for i in range(len(id_liked_recipes)):
        liked_recipes.append(Recipe.objects.filter(Id_Recipe=id_liked_recipes[i])[0])
        
    id_disliked_recipes = []
    for i in range(len(user_reactions)):
        if user_reactions[i].reaction == 'dislike':
            id_disliked_recipes.append(user_reactions[i].recipe_id)
    disliked_recipes = []
    for i in range(len(id_disliked_recipes)):
        disliked_recipes.append(Recipe.objects.filter(Id_Recipe=id_disliked_recipes[i])[0])

    # 2. Ништяки: Агрегированная статистика

    # Общее количество реакций и статистика лайков/дизлайков за один запрос
    reaction_stats = user_reactions.aggregate(
        total=Count('reaction'),
        likes=Count('reaction', filter=Q(reaction='like')),
        dislikes=Count('reaction', filter=Q(reaction='dislike'))
    )

    context = {
        # Объект CustomUser (доступны поля username, email, date_joined и т.д.)
        'user': user,
        'liked_recipes': liked_recipes,
        'disliked_recipes': disliked_recipes,
        # Передаем агрегированную статистику
        'reaction_stats': reaction_stats,
    }

    return render(request, 'settings/settings.html', context)

# def index(request):
#     import test_10_best
#     index_person = int(request.GET.get('user_id', 270))
#     count_top = int(request.GET.get('top', 5))
#     print(index_person, count_top)
#     df = test_10_best.start_fun(n=index_person, top=count_top)
#     # return render(request, 'a.html')
#     return HttpResponse(df.to_html())

# @login_required
def card(request, id):
    
    current_recipe = Recipe.objects.filter(
        Id_Recipe=id
    )
    for recipe in current_recipe:
        if recipe.Description.strip() == 'nan':
            recipe.Description = None
        # print(recipe.Description)
        temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        try:
            recipe.Images_recipe = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        except:
            recipe.Images_recipe = None
            
        # print(recipe.Steps_images)
        # temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        # if temp_global_link and temp_global_link in file_data_images_to_local:
        #     recipe.Image_path = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        # else:
        #     recipe.Image_path = 'images/not_image/not_image_recipe.png' # Путь по умолчанию
            
            
            
        temp_step = extract_data_from_string_2(recipe.Steps_text, single_item=False)
        # print('000', recipe.Steps_text)
        # print('111', temp_step)
        try:
            recipe.Steps_text = temp_step
        except:
            recipe.Steps_text = None
        
        # print(recipe.Tags, type(recipe.Ingredients))
        raw_list = ast.literal_eval(recipe.Ingredients.strip())
        # print(raw_list)
        ingredients = [[item[1], item[2]] for item in raw_list]
        # print(ingredients)
        recipe.Ingredients = ingredients
        
        temp_tags = ast.literal_eval(recipe.Tags.strip())
        recipe.Tags = temp_tags
        # print(temp_tags, type(temp_tags ))
        
        try:
            # Превращаем строку "[['0', 'url'], ...]" в список списков
            raw_steps_images = ast.literal_eval(recipe.Url_steps_images)
        except:
            raw_steps_images = []
        steps_images_paths = []
        for item in raw_steps_images:
            # Извлекаем URL (второй элемент подсписка)
            url_from_db = item[1] if isinstance(item, list) and len(item) > 1 else ""
            
            # Находим локальный путь через ваш глобальный словарь
            local_name = file_data_images_to_local.get(url_from_db)
            
            if local_name:
                steps_images_paths.append('images/images/' + local_name)
            # else:
            #     steps_images_paths.append('images/not_image/not_image_recipe.png')

        # Сохраняем результат обратно в объект рецепта (или в новую переменную)
        recipe.processed_steps_images = steps_images_paths
        
        
    return render(request, 'card_recipe/card_recipe.html', {"recipe":current_recipe})
    
    
    
    # card_recipe
    
    # template = loader.get_template('b.html')
    # df = pd.read_csv('dataset/data.csv')
    # df_id = df[df['id'] == id].iloc[0]
    # context = df_id.to_dict()
    # if context["Images_recipe"] != '[]':
    #     context['Images_recipe'] = ast.literal_eval(
    #         context['Images_recipe'])[0][1]
    #     # temp_image_recipe = context['Images_recipe']
    #     # print(df['id'])
    #     local_link = file_data_images_to_local.get(context['Images_recipe'])
    #     # print(local_link)
    #     if local_link != None:
    #         context['Images_recipe'] = 'images/images/'+local_link
    #     else:
    #         context['Images_recipe'] = 'images/not_image/not_image_recipe.png'
    # else:
    #     context['Images_recipe'] = 'images/not_image/not_image_recipe.png'
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
    # return HttpResponse(template.render(context, request))
    # return HttpResponse(f"<h1>Имя: {name}</h1>")

# def card(request):
#     return render(request, 'b.html')
#     template = loader.get_template('myapp/templates/b.html')
#     context = {'message': 'Привет, мир!'}
#     return HttpResponse(template.render(context, request))


def home(request):
    return redirect('whoami')


def whoami(request):
    # Получаем user_id из сессии
    # user_id = request.session.get('user_id')

    # # Получаем пользователя или None
    # user = None
    # if user_id:
    #     try:
    #         user = MyUser.objects.get(id=user_id)
    #     except MyUser.DoesNotExist:
    #         # Если пользователь не найден, очищаем сессию
    #         request.session.flush()

    # # Передаем в контекст
    # context = {
    #     'user': user,  # Полный объект пользователя или None
    #     'user_id': user.id if user else None,  # Только ID или None
    #     'is_authenticated': user is not None,  # Флаг авторизации
    #     'username': user.username if user else 'Гость',  # Имя или "Гость"
    #     'email': user.email if user else '',
    # }

    # user = request.user.is_authenticated
    context = {}
    if request.user.is_authenticated:
        context['user_id'] = request.user.id
        context['b'] = True
        context['username'] = request.user.username
        context['email'] = request.user.email
        context['phone_number'] = request.user.phone_number
    print(context)
    return render(request, 'home/test.html', context)


def logout_page(request):
    if request.user.is_authenticated:
        logout(request)

    # Получаем токен из cookies
    # token = request.COOKIES.get('session_token')

    # Проверяем, есть ли user_id в сессии Django
    # user_id = request.session.get('user_id')
    # print(token, user_id)
    # if user_id and token:
    #     try:
    #         # Находим пользователя
    #         user = MyUser.objects.get(id=user_id)
    #         print('пользователь найден')
    #         # Проверяем валидность сессии (опционально, но рекомендуется)
    #         if user.validate_session(token):
    #             user.logout()  # Вызываем метод logout модели

        # Если хотите строгую проверку, можно сделать так:
        # if user.session_token == token:
        #     user.logout()

        # except MyUser.DoesNotExist:
        #     pass  # Пользователь не найден, ничего не делаем

    # Очищаем сессию Django
    # request.session.flush()

    # Создаем ответ и удаляем куки
    # response = redirect('login')  # Перенаправляем на страницу входа
    # response.delete_cookie('session_token')

    # Если у вас есть другие куки для сессии, удалите их тоже:
    # response.delete_cookie('session_id')
    # response.delete_cookie('remember_me')

    return redirect('/login')


def login_page(request):
    if request.method == 'POST':
        # Получаем данные из формы
        email_input = request.POST.get('email_input')
        password_input = request.POST.get('password_input')
        remember_me = request.POST.get('remember_me')  # опционально

        # print(f'{email_input=}, {password_input=}')
        try:
            user = CustomUser.objects.get(email=email_input)
        except CustomUser.DoesNotExist:
            user = None
        if user:
            if user.check_password(password_input):
                login(request, user)
                print('УСПЕШНЫЙ ВХОД')
                return redirect('/')
            else:
                print('НЕВЕРНЫЙ ПАРОЛЬ')
        else:
            print('НЕТ ПОЛЬЗОВАТЕЛЯ')
        # print(user)
        return redirect('/login')

        # Валидация
        # errors = []

        # if not email_input:
        #     errors.append("Введите логин или email")
        # if not password_input:
        #     errors.append("Введите пароль")

        # Если есть ошибки валидации
        # if errors:
        #     return render(request, 'login/login.html', {
        #         'errors': errors,
        #         'username_input': username_input,  # Возвращаем логин
        #         # Пароль НЕ возвращаем!
        #         'remember_me': remember_me
        #     })
        # try:
        #     user = CustomUser.objects.get(email=email_input)
        # print(user.username, user.email, user.password_hash)
        # import secrets
        # import hashlib
        # salt = secrets.token_hex(16)  # 32 символа в hex
        # hash_obj = hashlib.sha256(f"{salt}{password_input}".encode())
        # hesh_password =  f"{salt}${hash_obj.hexdigest()}"
        # print(user.check_password('p'))
        # print(user.username, user.email, user.password)
        # if user.check_password(password_input):
        #     print('ТАКОЙ ПОЛЬЗОВАТЕЛЬ ЕСТЬ!!!')
        # 1. Создаем сессию в модели
        # token = user.create_session(remember=remember_me)

        # # 2. Сохраняем в Django session
        # request.session['user_id'] = user.id
        # request.session['session_token'] = token

        # # 3. Опционально: другие данные
        # request.session['username'] = user.username

        # # 4. Создаем response с redirect
        # response = redirect('/whoami')

        # # 5. Устанавливаем cookie для браузера
        # response.set_cookie('auth_token', token,
        #                     httponly=True, secure=True)

        #     return response
        # else:
        #     print('Неверный пароль')
        #     errors.append('Неверный пароль')
        #     return render(request, 'login/login.html', {
        #         'errors': errors,
        #         'email_input': email_input,  # Показываем что вводили
        #         'remember_me': remember_me
        #     })

        # except:
        #     print('[!] Такого пользователя нет')
        #     errors.append('Такого пользователя нет')
        #     return render(request, 'login/login.html', {
        #         'errors': errors,
        #         'email_input': email_input,  # Показываем что вводили
        #         'remember_me': remember_me
        #     })
        # Проверяем пользователя
        # Пример с обычной проверкой:
        # from django.contrib.auth import authenticate, login as auth_login

        # user = authenticate(
        #     request,
        #     username=username_input,  # или email, если так настроено
        #     password=password_input
        # )

        # if user is not None:
        #     # Успешный логин
        #     auth_login(request, user)

        #     # "Запомнить меня"
        #     if remember_me:
        #         request.session.set_expiry(60 * 60 * 24 * 30)  # 30 дней
        #     else:
        #         request.session.set_expiry(0)  # до закрытия браузера

        #     return redirect('home')  # или куда нужно
        # else:
        #     # Неправильные учетные данные
        #     errors.append("Неверный логин или пароль")

        #     # Возвращаем страницу с ошибкой
        #     return render(request, 'login/login.html', {
        #         'errors': errors,
        #         'username_input': username_input,  # Показываем что вводили
        #         'remember_me': remember_me
        #     })

    # GET запрос - показать пустую форму
    return render(request, 'login/login.html', {
        'username_input': '',
        'remember_me': False
    })
# def login(request):
#     return render(request, 'login/login.html')


def site_registration(request):
    return render(request, 'registration/registration.html')

# Только для теста, лучше использовать csrf_token в форме


def process_button(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')  # получаем значение
        print(f"Получен текст: {user_input}")
        # request.session['a'] = 'q'
        # request.session.set_expiry(3600)

        # Перенаправление если все ок
        return redirect('/')
        # return render(request, 'home/test.html', context={'t':user_input})
    return redirect('/')


def reg(request):
    if request.method == 'POST':
        name_input = request.POST.get('name_input')
        email_input = request.POST.get('email_input')
        password_input = request.POST.get('password_input')

        # Проверяем данные
        errors = []

        # if not name_input:
        #     errors.append("Имя обязательно")
        # if not email_input:
        #     errors.append("Email обязателен")
        # if len(password_input) < 6:
        #     errors.append("Пароль должен быть не менее 6 символов")

        # Если есть ошибки
        # if errors:
        #     # Возвращаем шаблон с сохраненными данными и ошибками
        #     return render(request, 'registration/registration.html', {
        #         'errors': errors,
        #         'name_input': name_input,  # Возвращаем введенные данные
        #         'email_input': email_input,
        #         # password обычно не возвращают из соображений безопасности
        #     })

        # Если все ок - продолжаем обработку
        print(f'[!] {name_input=}\n[!] {email_input=}\n[!] {password_input=}')
        # ... сохранение в БД
        CustomUser()
        new_user = CustomUser.objects.create_user(
            username=name_input,
            email=email_input,
            password=password_input
        )
        new_user.save()
        # new_user.set_password(password_input)
        # new_user.save()
        print('[!] Новый пользователь создан!')
        return redirect('/login')

    # GET запрос - пустая форма
    return render(request, 'registration/registration.html', {
        'name_input': '',
        'email_input': '',
    })


@csrf_exempt
@require_POST
@login_required
def handle_reaction(request):
    """Обработка лайков/дизлайков через AJAX"""
    try:
        data = json.loads(request.body)
        recipe_id = data.get('recipe_id')
        reaction = data.get('reaction')  # 'like', 'dislike', или null

        recipe = Recipe.objects.get(id=recipe_id)

        # Удаляем существующую реакцию, если reaction = null
        if not reaction:
            RecipeReaction.objects.filter(
                user=request.user,
                recipe=recipe
            ).delete()
            return JsonResponse({
                'status': 'removed',
                'like_count': recipe.recipereaction_set.filter(reaction='like').count(),
                'dislike_count': recipe.recipereaction_set.filter(reaction='dislike').count()
            })

        # Обновляем или создаем реакцию
        reaction_obj, created = RecipeReaction.objects.update_or_create(
            user=request.user,
            recipe=recipe,
            defaults={'reaction': reaction}
        )

        # Получаем количество реакций
        like_count = recipe.recipereaction_set.filter(reaction='like').count()
        dislike_count = recipe.recipereaction_set.filter(
            reaction='dislike').count()

        return JsonResponse({
            'status': 'created' if created else 'updated',
            'reaction': reaction,
            'like_count': like_count,
            'dislike_count': dislike_count
        })

    except Recipe.DoesNotExist:
        return JsonResponse({'error': 'Recipe not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@require_POST
@login_required
# @csrf_exempt # ВНИМАНИЕ: Для продакшена лучше использовать CSRF-токен в JS, а не @csrf_exempt
def react_to_recipe(request):
    """Обрабатывает AJAX-запросы на лайк/дизлайк рецепта."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=403)

    try:
        data = json.loads(request.body)
        recipe_id = data.get('recipe_id')
        action = data.get('action')  # 'like' или 'dislike'

        print(f"Received data: recipe_id={recipe_id}, action={action}")

        if action not in ['like', 'dislike']:
            return JsonResponse({'error': 'Invalid action'}, status=400)

        recipe = Recipe.objects.get(id=recipe_id)
        user = request.user

        # 1. Проверяем, существует ли предыдущая реакция
        existing_reaction = RecipeReaction.objects.filter(
            user=user, recipe=recipe).first()

        if existing_reaction:
            if existing_reaction.reaction == action:
                # Нажали ту же кнопку -> Удаляем реакцию (none)
                existing_reaction.delete()
                reaction = 'none'
            else:
                # Нажали противоположную кнопку -> Обновляем реакцию
                existing_reaction.reaction = action
                existing_reaction.save()
                reaction = action
        else:
            # Новая реакция
            RecipeReaction.objects.create(
                user=user, recipe=recipe, reaction=action)
            reaction = action

        # 2. Получаем новые счетчики
        like_count = RecipeReaction.objects.filter(recipe=recipe, reaction='like').count() + \
            Recipe.objects.filter(Id_Recipe=recipe_id).first().Likes
        dislike_count = RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count() + \
            Recipe.objects.filter(Id_Recipe=recipe_id).first().Dislikes

        return JsonResponse({
            'status': 'success',
            'reaction': reaction,
            'like_count': like_count,
            'dislike_count': dislike_count
        })

    except Recipe.DoesNotExist:
        return JsonResponse({'error': 'Recipe not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def recipe_list(request):
    if not request.user.is_authenticated:
        return redirect('/login')
    
    cur_per_id = request.user.id
    
    # 1. Параметры пагинации
    page = int(request.GET.get('page', 1))
    limit = 10  
    
    # 2. Получаем рекомендации
    # ВАЖНО: get_recommendations_for_user возвращает QuerySet из модели Recipe
    # Увеличим n_top, чтобы хватило на несколько страниц или на текущий лимит
    recommended_recipes_qs = get_recommendations_for_user(cur_per_id, n_top=50)

    # 3. Применяем пагинацию к полученному списку рекомендаций
    start_idx = (page - 1) * limit
    end_idx = page * limit
    
    # Так как это QuerySet, мы можем использовать слайсинг
    recipes = recommended_recipes_qs[start_idx:end_idx]
    
    # Проверка на наличие следующей страницы
    has_next = recommended_recipes_qs.count() > end_idx

    # 4. Дополнительная обработка каждого рецепта для шаблона
    for recipe in recipes:
        # Счётчики лайков (БД + статика из модели)
        recipe.like_count = RecipeReaction.objects.filter(recipe_id=recipe.Id_Recipe, reaction='like').count() + \
            (recipe.Likes if recipe.Likes else 0)
        
        recipe.dislike_count = RecipeReaction.objects.filter(recipe_id=recipe.Id_Recipe, reaction='dislike').count() + \
            (recipe.Dislikes if recipe.Dislikes else 0)

        # Проверка реакции текущего пользователя
        user_reaction = RecipeReaction.objects.filter(
            user_id=cur_per_id,
            recipe_id=recipe.Id_Recipe
        ).first()
        recipe.user_reaction = user_reaction.reaction if user_reaction else None

        # Обработка пути к изображению через ваш JSON-конфиг
        temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        if temp_global_link and temp_global_link in file_data_images_to_local:
            recipe.Image_path = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        else:
            recipe.Image_path = 'images/not_image/not_image_recipe.png' # Путь по умолчанию

    # 5. AJAX-обработка (бесконечный скролл)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('home/recipe_cards_partial.html', {'recipes': recipes}, request=request)
        return JsonResponse({'html': html, 'has_next': has_next})

    # Основной рендер страницы
    return render(request, 'home/home3.html', {
        'recipes': recipes, 
        'has_next': has_next
    })


def test(request):
    # recipes = Recipe.objects.all()
    recipes = Recipe.objects.filter(Id_Recipe=0)
    print(type(recipes), recipes)
    return render(request, 'test/test.html', {'data': recipes})

    # {% for article in data %}
    # {{ article.Id_Recipe }}
    # {{ article.Name_recipe }}
    # {{ article.URL }}
    # {% endfor %}




def search_recipes(request):
    query = request.GET.get('q', '').strip()
    max_time = request.GET.get('max_time', '')
    selected_type = request.GET.get('type_recipe', '')
    
    # Собираем фильтры через объект Q для гибкости
    filters = Q()
    
    if query:
        # icontains делает поиск нечувствительным к регистру и ищет часть слова
        # Мы объединяем условия через ИЛИ (|), чтобы искать везде сразу
        filters &= (
            Q(Name_recipe__icontains=query) | 
            Q(Description__icontains=query) |
            Q(Tags__icontains=query)
        )

    if max_time:
        try:
            filters &= Q(Cooking_time__lte=int(max_time))
        except ValueError:
            pass
        
    if selected_type:
        filters &= Q(Type_recipe=selected_type)

    # Выполняем запрос с примененными фильтрами
    # distinct() нужен, чтобы избежать дублей, если слово нашлось и в тегах, и в имени
    queryset = Recipe.objects.filter(filters).distinct()
    
    # Ограничиваем выборку для производительности
    recipes = queryset[:40]

    # Обработка данных для отображения (логика из вашего файла)
    for recipe in recipes:
        recipe.like_count = RecipeReaction.objects.filter(recipe=recipe, reaction='like').count() + (recipe.Likes or 0)
        recipe.dislike_count = RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count() + (recipe.Dislikes or 0)
        
        if request.user.is_authenticated:
            react = RecipeReaction.objects.filter(user=request.user, recipe=recipe).first()
            recipe.user_reaction = react.reaction if react else None
        
        # Обработка пути к изображению через ваш JSON-конфиг
        temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        if temp_global_link and temp_global_link in file_data_images_to_local:
            recipe.Image_path = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        else:
            recipe.Image_path = 'images/not_image/not_image_recipe.png' # Путь по умолчанию
        
        # Безопасный парсинг тегов для шаблона
        try:
            recipe.Tags_list = ast.literal_eval(recipe.Tags.strip()) if recipe.Tags else []
        except:
            recipe.Tags_list = []

    # Получаем список всех типов для выпадающего списка в фильтре
    all_types = Recipe.objects.values_list('Type_recipe', flat=True).distinct().exclude(Type_recipe__isnull=True)

    return render(request, 'search/search.html', {
        'recipes': recipes,
        'query': query,
        'all_types': all_types,
        'selected_type': selected_type,
        'max_time': max_time
    })
    
    
def fridge_page(request):
    # Пример списка продуктов
    ingredients = [
        "Авокадо", "Баранина", "Баклажан", "Бекон", "Виноград", "Говядина", "Грибы",
        "Дрожжи", "Йогурт", "Капуста", "Картофель", "Курица", "Лук", "Макароны",
        "Молоко", "Мёд", "Огурцы", "Орехи", "Перец", "Помидоры", "Рис", "Рыба",
        "Сыр", "Сметана", "Творог", "Тыква", "Укроп", "Фасоль", "Фарш", "Хлеб",
        "Чеснок", "Шоколад", "Яблоки", "Яйца"
    ]
    ingredients.sort()
    
    grouped = defaultdict(list)
    for ing in ingredients:
        grouped[ing[0].upper()].append(ing)
        
    return render(request, 'fridge/fridge.html', {
        'grouped_ingredients': dict(sorted(grouped.items()))
    })