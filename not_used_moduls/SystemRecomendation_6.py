import os
import sys
import django
import numpy as np
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- DJANGO INIT -------------------

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from myapp.models import Recipe, RecipeReaction

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^а-яa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def preprocess_recipes(df):
    # Парсим строку в объект Python
    df["parsed_ingredients"] = df["Ingredients"].apply(safe_parse)

    # Исправление: Если внутри списка оказались другие списки, выравниваем их
    def flatten_ingredients(ing_list):
        flat = []
        for item in ing_list:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    df["parsed_ingredients"] = df["parsed_ingredients"].apply(flatten_ingredients)

    # Дальнейшая обработка строк для TF-IDF
    df["ingredients_str"] = df["parsed_ingredients"].apply(
        lambda lst: " ".join([normalize_text(str(i)) for i in lst])
    )

    df["text_blob"] = (
        df["Name_recipe"].fillna("") + " " +
        df["Description"].fillna("") + " " +
        df["ingredients_str"]
    )

    return df

from functools import lru_cache

@lru_cache(maxsize=1)
def build_tfidf():
    recipes = Recipe.objects.all().values()
    df = pd.DataFrame(list(recipes))
    df = preprocess_recipes(df)

    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(df["text_blob"])
    return df, matrix

def fridge_boost(recipe_ingredients, fridge_items):
    """
    Улучшенная функция буста: теперь она нормализует каждый ингредиент перед сравнением.
    """
    if not fridge_items:
        return 1.0

    # Приводим всё к нижнему регистру и чистим от пробелов для точного совпадения
    recipe_set = set(normalize_text(str(i)) for i in recipe_ingredients if i)
    fridge_set = set(normalize_text(str(i)) for i in fridge_items if i)

    # Находим пересечение
    overlap = len(recipe_set & fridge_set)
    if overlap == 0:
        return 0.05  # Сильно штрафуем рецепты без продуктов из холодильника
    
    ratio = overlap / max(len(recipe_set), 1)
    return 10.0 + (ratio * 50.0) # Даем гигантский отрыв рецептам с совпадениями

def get_recommendations(user_id, n_top=5, fridge_ingredients=None, randomness=0.2):
    df, tfidf = build_tfidf()
    
    # 1. Получаем реакции
    interactions = RecipeReaction.objects.filter(user_id=user_id)
    liked = list(interactions.filter(reaction="like").values_list("recipe_id", flat=True))
    disliked = list(interactions.filter(reaction="dislike").values_list("recipe_id", flat=True))
    
    id_to_idx = {rid: i for i, rid in enumerate(df["id"])}

    # 2. Базовый скор
    if liked:
        vectors = [tfidf[id_to_idx[rid]] for rid in liked if rid in id_to_idx]
        if vectors:
            from scipy.sparse import vstack
            user_vec = vstack(vectors).mean(axis=0)
            user_vec = np.asarray(user_vec)
            scores = cosine_similarity(user_vec, tfidf)[0]
        else:
            scores = np.zeros(len(df))
    else:
        # Если лайков нет, даем всем одинаковый начальный скор
        scores = np.ones(len(df)) * 0.1 

    # 3. Учет холодильника (ПРИНУДИТЕЛЬНЫЙ)
    if fridge_ingredients:
        boosts = df["parsed_ingredients"].apply(
            lambda x: fridge_boost(x, fridge_ingredients)
        )
        
        # Если в рецепте НЕТ ничего из холодильника, мы его почти обнуляем
        # Проверяем, был ли overlap (в вашей функции это возврат 1.0)
        final_boosts = np.where(boosts.values > 1.0, boosts.values * 10, 0.001)
        scores = scores * final_boosts

    # 4. Штраф за дизлайки (умножаем на 0, чтобы исключить или сильно опустить)
    for rid in disliked:
        if rid in id_to_idx:
            scores[id_to_idx[rid]] = 0.0

    df["score"] = scores

    # 5. Фильтрация и Сортировка
    seen = set(liked) | set(disliked)
    candidates = df[~df["id"].isin(seen)].copy()
    
    # Сортируем строго по score
    candidates = candidates.sort_values("score", ascending=False)

    # 6. Рандом (исправленная логика)
    if randomness > 0 and len(candidates) > n_top:
        n_fixed = int(n_top * (1 - randomness))
        top_part = candidates.iloc[:n_fixed]
        random_part = candidates.iloc[n_fixed : n_top * 3].sample(n=n_top - n_fixed)
        final_df = pd.concat([top_part, random_part])
    else:
        final_df = candidates.head(n_top)

    # 7. ВАЖНО: Сохранение порядка в Django
    final_ids = final_df["id"].tolist()
    if not final_ids:
        return Recipe.objects.none()
        
    preserved_order = django.db.models.Case(
        *[django.db.models.When(pk=pk, then=pos) for pos, pk in enumerate(final_ids)]
    )
    
    return Recipe.objects.filter(id__in=final_ids).order_by(preserved_order)

if __name__ == "__main__":
    try:
        TEST_USER_ID = 9 
        # Добавьте в список то, что точно есть в ваших рецептах (например, "курица")
        my_fridge = ["картофель", "лук", 'Курица'] 
        
        print(f"\n--- ИСТОРИЯ ПОЛЬЗОВАТЕЛЯ ID {TEST_USER_ID} ---")
        reactions = RecipeReaction.objects.filter(user_id=TEST_USER_ID)
        for r in reactions:
            recipe = Recipe.objects.get(id=r.recipe_id)
            print(f"{'👍' if r.reaction=='like' else '👎'} {recipe.Name_recipe}")

        print(f"\n--- РЕКОМЕНДАЦИИ (Холодильник: {my_fridge}) ---")
        recs = get_recommendations(
            TEST_USER_ID, 
            n_top=7, 
            fridge_ingredients=my_fridge,
            randomness=0.2
        )
        
        for r in recs:
            # Для отладки выведем ингредиенты
            print(f"ID: {r.id} | {r.Name_recipe}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()