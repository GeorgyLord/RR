'''
МОДУЛЬ ГЛАВНОЙ ЛОГИКИ СИСТЕМЫ
'''

# import csv
import os
from django.conf import settings
# from django.http import HttpResponse
from collections import defaultdict
import hashlib
import uuid
# import pandas as pd
# from django.template import loader
# from test_10_best import start_fun
# from SystemRecomendation_3 import get_recommendations_for_user
# from SystemRecomendation_5 import get_recommendations_for_user
from SystemRecomendation_7 import get_recommendations_for_user
# from SystemRecomendation_2 import start_fun
import ast
import json
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
# from django.db.models.functions import Lower
from django.contrib import messages
# from .forms import RegisterForm, LoginForm
from .models import RecipeReaction, Recipe
from .models import CustomUser
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Count
from django.http import JsonResponse
# from django.db.models import Case, When, Value, IntegerField
from django.template.loader import render_to_string
from .forms import RecipeForm
# from .models import MyUser
from django.db.models import Max


# class MyUser():
#     def __init__(self):
#         pass

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
    Безопасно парсит строку вида "[['0', 'данные1'], ['1', 'данные2'], ...]" и извлекает данные.
    
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
    
    # Получаем реакции сразу с загруженными рецептами
    user_reactions = RecipeReaction.objects.filter(user=user).select_related('recipe')
    
    liked_recipes = [
        reaction.recipe for reaction in user_reactions if reaction.reaction == 'like'
    ]
    
    disliked_recipes = [
        reaction.recipe for reaction in user_reactions if reaction.reaction == 'dislike'
    ]

    # Статистика
    reaction_stats = user_reactions.aggregate(
        total=Count('reaction'),
        likes=Count('reaction', filter=Q(reaction='like')),
        dislikes=Count('reaction', filter=Q(reaction='dislike'))
    )

    # Созданные рецепты с подсчетом лайков
    created_recipes_qs = user.created_recipes.all()
    created_recipes_list = []
    
    for recipe in created_recipes_qs:
        # Считаем сумму статических лайков и реакций
        real_likes = recipe.Likes + RecipeReaction.objects.filter(recipe=recipe, reaction='like').count()
        real_dislikes = recipe.Dislikes + RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count()
        
        recipe.total_likes = real_likes
        recipe.total_dislikes = real_dislikes
        created_recipes_list.append(recipe)

    context = {
        'user': user,
        'liked_recipes': liked_recipes,
        'disliked_recipes': disliked_recipes,
        'reaction_stats': reaction_stats,
        'created_recipes': created_recipes_list,
    }
    return render(request, 'settings/settings.html', context)


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
        
        
    user_reaction = None
    if request.user.is_authenticated:
        # Ищем реакцию в БД. Обратите внимание: recipe=recipe_obj (связь идет по системному id, Django сам это сделает)
        recipe_obj = current_recipe.first()
        reaction_obj = RecipeReaction.objects.filter(
            user=request.user,
            recipe=recipe_obj 
        ).first()
        if reaction_obj:
            user_reaction = reaction_obj.reaction

    # Передаем user_reaction в контекст
    return render(request, 'card_recipe/card_recipe.html', {
        "recipe": current_recipe, 
        "user_reaction": user_reaction,
        "count_liks":RecipeReaction.objects.filter(recipe=recipe_obj, reaction='like').count()+current_recipe.first().Likes,
        "count_disliks":RecipeReaction.objects.filter(recipe=recipe_obj, reaction='dislike').count()+current_recipe.first().Dislikes,
    })
    



def home(request):
    return redirect('whoami')


def whoami(request):

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

    return redirect('/login')


def login_page(request):
    if request.method == 'POST':
        # Получаем данные из формы
        email_input = request.POST.get('email_input', '').strip()
        password_input = request.POST.get('password_input', '')
        remember_me = request.POST.get('remember_me')
        
        errors = {}
        
        # Валидация полей
        if not email_input:
            errors['email'] = 'Пожалуйста, введите email'
        elif '@' not in email_input:
            errors['email'] = 'Введите корректный email адрес'
        
        if not password_input:
            errors['password'] = 'Пожалуйста, введите пароль'
        
        # Проверка пользователя только если нет ошибок валидации
        if not errors:
            try:
                user = CustomUser.objects.get(email=email_input)
                
                if user.check_password(password_input):
                    login(request, user)
                    
                    # Обработка "Запомнить меня"
                    if remember_me:
                        request.session.set_expiry(30 * 24 * 60 * 60)  # 30 дней
                    else:
                        request.session.set_expiry(0)  # До закрытия браузера
                    
                    messages.success(request, 'Вы успешно вошли в систему!')
                    return redirect('/')
                else:
                    errors['general'] = 'Неверный email или пароль'
            except CustomUser.DoesNotExist:
                errors['general'] = 'Неверный email или пароль'
        else:
            # Сохраняем введенные данные для повторного заполнения формы
            request.session['form_data'] = {
                'email_input': email_input,
                'remember_me': bool(remember_me)
            }
        
        # Если есть ошибки, показываем форму снова
        return render(request, 'login/login.html', {
            'errors': errors,
            'form_data': {
                'email_input': email_input,
                'remember_me': bool(remember_me)
            }
        })

    # GET запрос - показать пустую форму
    # Проверяем, есть ли сохраненные данные в сессии
    form_data = request.session.pop('form_data', None)
    
    return render(request, 'login/login.html', {
        'errors': {},
        'form_data': form_data or {
            'email_input': '',
            'remember_me': False
        }
    })


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

        errors = []

        # Проверка: Существует ли уже такой логин?
        if CustomUser.objects.filter(username=name_input).exists():
            errors.append("Пользователь с таким именем уже существует")

        # Проверка: Существует ли уже такой email? (Опционально, но рекомендуется)
        if CustomUser.objects.filter(email=email_input).exists():
            errors.append("Этот Email уже зарегистрирован")

        # Если есть ошибки, возвращаем их на страницу регистрации
        if errors:
            return render(request, 'registration/registration.html', {
                'errors': errors,
                'name_input': name_input,   # Возвращаем введенные данные, чтобы не вводить заново
                'email_input': email_input,
            })

        try:
            # Создание пользователя
            # Примечание: create_user автоматически хеширует пароль и сохраняет объект
            new_user = CustomUser.objects.create_user(
                username=name_input,
                email=email_input,
                password=password_input
            )
            
            # Строка new_user.save() не нужна, create_user уже сохранил его
            
            print('[!] Новый пользователь создан!')
            messages.success(request, 'Регистрация прошла успешно! Теперь войдите.')
            return redirect('/login')
            
        except Exception as e:
            # Если произошла другая ошибка базы данных
            return render(request, 'registration/registration.html', {
                'errors': [f"Ошибка регистрации: {str(e)}"],
                'name_input': name_input,
                'email_input': email_input,
            })

    # GET запрос - пустая форма
    return render(request, 'registration/registration.html', {
        'name_input': '',
        'email_input': '',
        'errors': []
    })


@csrf_exempt
@require_POST
@login_required
def handle_reaction(request):
    # print(111)
    """Обработка лайков/дизлайков через AJAX"""
    try:
        data = json.loads(request.body)
        recipe_id = data.get('recipe_id')
        reaction = data.get('reaction')  # 'like', 'dislike', или null
        # print(recipe_id)
        recipe = Recipe.objects.get(Id_Recipe=recipe_id)

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
def react_to_recipe(request):
    # print(11)
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

        recipe = Recipe.objects.get(Id_Recipe=recipe_id)
        user = request.user

        # Проверяем, существует ли предыдущая реакция
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

        # print(RecipeReaction.objects.filter(recipe=recipe, reaction='like').count(), Recipe.objects.filter(Id_Recipe=recipe_id).first().Likes)
        # Получаем новые счетчики
        like_count = RecipeReaction.objects.filter(recipe=recipe, reaction='like').count() + \
            Recipe.objects.filter(Id_Recipe=recipe_id).first().Likes
        dislike_count = RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count() + \
            Recipe.objects.filter(Id_Recipe=recipe_id).first().Dislikes

        # print(RecipeReaction.objects.filter(recipe=recipe, reaction='like').count(), Recipe.objects.filter(Id_Recipe=recipe_id).first().Likes)

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
    page = int(request.GET.get('page', 1))
    limit = 10  
    
    # Получаем рекомендации (в QuerySet)
    recommended_recipes_qs = get_recommendations_for_user(cur_per_id, n_top=4000)

    # Пагинация
    start_idx = (page - 1) * limit
    end_idx = page * limit
    recipes = recommended_recipes_qs[start_idx:end_idx]
    has_next = recommended_recipes_qs.count() > end_idx

    # Получаем список ID рецептов на текущей странице для оптимизации запросов
    # (чтобы не делать запрос в цикле для каждого рецепта)
    current_page_recipe_ids = [r.id for r in recipes]

    # Получаем реакции пользователя для ВСЕХ рецептов на странице одним запросом
    user_reactions_map = {}
    if request.user.is_authenticated:
        reactions = RecipeReaction.objects.filter(
            user=request.user,
            recipe_id__in=current_page_recipe_ids
        ).values('recipe_id', 'reaction')
        
        # Создаем словарь {system_id: reaction}
        for r in reactions:
            user_reactions_map[r['recipe_id']] = r['reaction']

    # Обработка рецептов
    for recipe in recipes:
        # ПОДСЧЕТ ЛАЙКОВ
        # filter(recipe=recipe) использует системный id объекта recipe, это правильно.
        current_likes_in_db = RecipeReaction.objects.filter(recipe=recipe, reaction='like').count()
        current_dislikes_in_db = RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count()
        
        # Складываем со статикой
        recipe.like_count = recipe.Likes + current_likes_in_db
        recipe.dislike_count = recipe.Dislikes + current_dislikes_in_db

        # РЕАКЦИЯ ПОЛЬЗОВАТЕЛЯ
        # Берем из предварительно загруженного словаря по системному id
        recipe.user_reaction = user_reactions_map.get(recipe.id)

        # Обработка картинок
        temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        if temp_global_link and temp_global_link in file_data_images_to_local:
            recipe.Image_path = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        else:
            recipe.Image_path = 'images/not_image/not_image_recipe.png'
            
        try:
            if isinstance(recipe.Tags, str):
                recipe.Tags_list = ast.literal_eval(recipe.Tags.strip())
            else:
                recipe.Tags_list = recipe.Tags
        except (ValueError, SyntaxError):
            recipe.Tags_list = []

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('home/recipe_cards_partial.html', {'recipes': recipes}, request=request)
        return JsonResponse({'html': html, 'has_next': has_next})

    return render(request, 'home/home3.html', {
        'recipes': recipes, 
        'has_next': has_next
    })


def test(request):
    # recipes = Recipe.objects.all()
    recipes = Recipe.objects.filter(Id_Recipe=0)
    print(type(recipes), recipes)
    return render(request, 'test/test.html', {'data': recipes})




def search_recipes(request):
    query = request.GET.get('q', '').strip()
    max_time = request.GET.get('max_time', '')
    selected_type = request.GET.get('type_recipe', '')
    # Новое поле: ингредиенты
    ingredients_query = request.GET.get('ingredients', '').strip()
    
    filters = Q()
    
    # Основной поиск по тексту
    if query:
        filters &= (
            Q(Name_recipe__icontains=query) | 
            Q(Description__icontains=query) |
            Q(Tags__icontains=query)
        )

    # Фильтр по времени
    if max_time:
        try:
            filters &= Q(Cooking_time__lte=int(max_time))
        except ValueError:
            pass
        
    # Фильтр по типу
    if selected_type:
        filters &= Q(Type_recipe=selected_type)

    # ФИЛЬТР ПО ИНГРЕДИЕНТАМ
    if ingredients_query:
        # Разбиваем строку "яйца, молоко" на список ['яйца', 'молоко']
        ing_list = [x.strip() for x in ingredients_query.split(',') if x.strip()]
        
        for ing in ing_list:
            # Каждый ингредиент должен быть в поле Ingredients
            # Используем icontains, так как Ingredients у вас хранится как строка
            filters &= Q(Ingredients__icontains=ing)

    # Выполняем запрос
    queryset = Recipe.objects.filter(filters).distinct()
    recipes = queryset[:40]

    for recipe in recipes:
        # (Оставляем вашу логику подсчета лайков без изменений)
        recipe.like_count = RecipeReaction.objects.filter(recipe=recipe, reaction='like').count() + recipe.Likes
        recipe.dislike_count = RecipeReaction.objects.filter(recipe=recipe, reaction='dislike').count() + recipe.Dislikes
    
        if request.user.is_authenticated:
            user_react = RecipeReaction.objects.filter(recipe=recipe, user=request.user).first()
            recipe.user_reaction = user_react.reaction if user_react else None
        
        temp_global_link = extract_url_from_string(recipe.Url_images_recipe)
        if temp_global_link and temp_global_link in file_data_images_to_local:
            recipe.Image_path = 'images/images/' + file_data_images_to_local.get(temp_global_link)
        else:
            recipe.Image_path = 'images/not_image/not_image_recipe.png'
        
        try:
            recipe.Tags_list = ast.literal_eval(recipe.Tags.strip()) if recipe.Tags else []
        except:
            recipe.Tags_list = []

    all_types = Recipe.objects.values_list('Type_recipe', flat=True).distinct().exclude(Type_recipe__isnull=True)

    return render(request, 'search/search2.html', {
        'recipes': recipes,
        'query': query,
        'all_types': all_types,
        'selected_type': selected_type,
        'max_time': max_time,
        'ingredients_query': ingredients_query, # Возвращаем введенные ингредиенты обратно в шаблон
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
    
    
def about_page(request):
    return render(request, 'about/about.html', {})

def add_recipe_page(request):
    if request.method == 'POST':
        form = RecipeForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                recipe = form.save(commit=False)
                
                # Пути к файлам
                json_path = os.path.join(settings.BASE_DIR, 'images_to_local.json')
                save_dir = os.path.join(settings.BASE_DIR, 'static', 'images', 'images')
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # ЗАГРУЗКА JSON (Один раз в начале)
                local_images_map = {}
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            local_images_map = json.load(f)
                    except json.JSONDecodeError:
                        local_images_map = {}

                # Вспомогательная функция для сохранения фото
                def process_image(image_file):
                    if not image_file: 
                        return None
                    
                    # Генерируем имена
                    ext = os.path.splitext(image_file.name)[1]
                    random_name = f"img_{uuid.uuid4()}{ext}" # Виртуальное имя
                    hashed_name = hashlib.md5(random_name.encode()).hexdigest() + ext # Реальное имя файла
                    
                    # Сохраняем файл
                    file_path = os.path.join(save_dir, hashed_name)
                    with open(file_path, 'wb+') as destination:
                        for chunk in image_file.chunks():
                            destination.write(chunk)
                    
                    # Добавляем в карту
                    local_images_map[random_name] = hashed_name
                    return random_name

                # ID RECIPE
                max_id_dict = Recipe.objects.aggregate(Max('Id_Recipe'))
                max_id = max_id_dict['Id_Recipe__max']
                recipe.Id_Recipe = 0 if max_id is None else int(max_id) + 1

                # ГЛАВНОЕ ФОТО
                main_image = request.FILES.get('main_image')
                main_img_name = process_image(main_image)
                
                if main_img_name:
                    recipe.Url_images_recipe = str([['0', main_img_name]])
                    recipe.Images_recipe = str([['0', main_img_name]])
                else:
                    recipe.Url_images_recipe = "[]"
                    recipe.Images_recipe = "[]" # Оставляем пустым или заполняем по желанию

                # ШАГИ ПРИГОТОВЛЕНИЯ ТЕКСТ + ФОТО
                steps_json = request.POST.get('steps', '[]')
                
                formatted_steps_text = []   # Для Steps_text
                formatted_steps_urls = []   # Для Url_steps_images
                
                try:
                    steps_data = json.loads(steps_json)
                    step_write_index = 0 # Индекс, который пойдет в БД (0, 1, 2 и др)
                    
                    for step in steps_data:
                        text = step.get('text', '').strip()
                        # Берем stepId из JSON, чтобы найти соответствующий файл в request.FILES
                        step_id_from_js = step.get('stepId') 
                        
                        if text:
                            # Текст
                            formatted_steps_text.append([str(step_write_index), text])
                            
                            # Картинка
                            # Ищем файл с именем step_image_{ID_из_JS}
                            img_key = f"step_image_{step_id_from_js}"
                            step_file = request.FILES.get(img_key)
                            
                            step_img_name = process_image(step_file)
                            
                            if step_img_name:
                                # Добавляем в список: ['0', 'random_name.jpg']
                                formatted_steps_urls.append([str(step_write_index), step_img_name])
                            
                            step_write_index += 1
                    
                    recipe.Steps_text = str(formatted_steps_text)
                    recipe.Url_steps_images = str(formatted_steps_urls)
                    recipe.Steps_images = str(formatted_steps_urls)
                    
                except Exception as e:
                    print(f"Ошибка шагов: {e}")
                    recipe.Steps_text = "[]"
                    recipe.Url_steps_images = "[]"

                # СОХРАНЕНИЕ JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(local_images_map, f, ensure_ascii=False, indent=4)

                # ОСТАЛЬНЫЕ ПОЛЯ
                # Tags
                raw_tags = form.cleaned_data.get('Tags', '')
                if raw_tags:
                    tags_list = [tag.strip() for tag in raw_tags.split(',') if tag.strip()]
                    recipe.Tags = str(tags_list)
                else:
                    recipe.Tags = "[]"

                # Ingredients
                ingredients_text = form.cleaned_data.get('Ingredients', '')
                ingredients_lines = [line.strip() for line in ingredients_text.split('\n') if line.strip()]
                formatted_ingredients = []
                for index, line in enumerate(ingredients_lines):
                    name = line; quantity = ""
                    if ' - ' in line: parts = line.split(' - ', 1); name, quantity = parts[0].strip(), parts[1].strip()
                    elif '-' in line: parts = line.split('-', 1); name, quantity = parts[0].strip(), parts[1].strip()
                    formatted_ingredients.append([str(index), name, quantity])
                recipe.Ingredients = str(formatted_ingredients)
                recipe.Count_ingredients = len(formatted_ingredients)

                # Meta
                recipe.Author = request.user.username if request.user.is_authenticated else "Guest"
                recipe.Number_page = 0
                recipe.Likes = 0; recipe.Dislikes = 0; recipe.Safes = 0
                recipe.URL = f"/recipe/"

                # Сохранение рецепта
                recipe.save()
                recipe.URL = f"/recipe/{recipe.Id_Recipe}/"
                recipe.save()
                
                if request.user.is_authenticated:
                    request.user.created_recipes.add(recipe)
                
                messages.success(request, 'Рецепт успешно добавлен!')
                return redirect('/')
                
            except Exception as e:
                print(f"Global Error: {e}")
                messages.error(request, f'Ошибка: {str(e)}')
        else:
            messages.error(request, f'Ошибка формы: {form.errors}')
    else:
        form = RecipeForm()
    
    return render(request, 'add_recipe/add_recipe.html', {'form': form})


