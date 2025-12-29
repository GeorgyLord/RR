import os
import sys
import django
import numpy as np
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

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
    """Подготовка текстовых данных и ингредиентов."""
    df["parsed_ingredients"] = df["Ingredients"].apply(safe_parse)

    def flatten_ingredients(ing_list):
        flat = []
        for item in ing_list:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    df["parsed_ingredients"] = df["parsed_ingredients"].apply(flatten_ingredients)

    df["ingredients_str"] = df["parsed_ingredients"].apply(
        lambda lst: " ".join([normalize_text(str(i)) for i in lst])
    )

    df["text_blob"] = (
        df["Name_recipe"].fillna("") + " " +
        df["Description"].fillna("") + " " +
        df["ingredients_str"]
    )
    return df

@lru_cache(maxsize=1)
def build_tfidf():
    """Сборка матрицы признаков на основе всех рецептов."""
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
    """Расчет бонуса за совпадение ингредиентов из холодильника."""
    if not fridge_items:
        return 1.0

    recipe_set = set(normalize_text(str(i)) for i in recipe_ingredients if i)
    fridge_set = set(normalize_text(str(i)) for i in fridge_items if i)

    overlap = len(recipe_set & fridge_set)
    if overlap == 0:
        return 0.05  # Штраф за отсутствие совпадений
    
    ratio = overlap / max(len(recipe_set), 1)
    return 10.0 + (ratio * 50.0) # Сильный приоритет при совпадении

def get_recommendations(user_id, n_top=5, fridge_ingredients=None, randomness=0.2):
    """Основная логика системы рекомендаций."""
    df, tfidf = build_tfidf()
    
    # 1. Получаем реакции пользователя
    interactions = RecipeReaction.objects.filter(user_id=user_id)
    # Используем Id_Recipe из связанных объектов
    liked = list(interactions.filter(reaction="like").values_list("recipe__Id_Recipe", flat=True))
    disliked = list(interactions.filter(reaction="dislike").values_list("recipe__Id_Recipe", flat=True))
    
    # Привязываем индекс матрицы к полю Id_Recipe
    id_to_idx = {rid: i for i, rid in enumerate(df["Id_Recipe"])}

    # 2. Расчет базового скора (Content-Based)
    if liked:
        vectors = [tfidf[id_to_idx[rid]] for rid in liked if rid in id_to_idx]
        if vectors:
            from scipy.sparse import vstack
            user_vec = vstack(vectors).mean(axis=0)
            user_vec = np.asarray(user_vec)
            scores = cosine_similarity(user_vec, tfidf)[0]
        else:
            scores = np.ones(len(df))
    else:
        scores = np.ones(len(df)) * 0.1 

    # 3. Учет холодильника
    if fridge_ingredients:
        boosts = df["parsed_ingredients"].apply(
            lambda x: fridge_boost(x, fridge_ingredients)
        )
        final_boosts = np.where(boosts.values > 1.0, boosts.values * 10, 0.001)
        scores = scores * final_boosts

    # 4. Штраф за дизлайки
    for rid in disliked:
        if rid in id_to_idx:
            scores[id_to_idx[rid]] = 0.0

    df["score"] = scores
    
    # 5. Фильтрация и Сортировка
    seen = set(liked) | set(disliked)
    # Фильтруем по Id_Recipe
    candidates = df[~df["Id_Recipe"].isin(seen)].copy()
    candidates = candidates.sort_values("score", ascending=False)

    # --- РАСЧЕТ ПОКАЗАТЕЛЕЙ (Абсолютный и Относительный) ---
    if not candidates.empty:
        def get_absolute_match(recipe_ing):
            if not fridge_ingredients: return 0
            r_set = set(normalize_text(str(i)) for i in recipe_ing if i)
            f_set = set(normalize_text(str(i)) for i in fridge_ingredients if i)
            intersect = len(r_set & f_set)
            return int((intersect / len(r_set)) * 100) if r_set else 0

        candidates["abs_fridge_match"] = candidates["parsed_ingredients"].apply(get_absolute_match)

        max_s = candidates["score"].max()
        min_s = candidates["score"].min()
        if max_s > min_s:
            candidates["match_percentage"] = candidates["score"].apply(
                lambda s: int(60 + (40 * (s - min_s) / (max_s - min_s)))
            )
        else:
            candidates["match_percentage"] = 80
    else:
        candidates["abs_fridge_match"] = 0
        candidates["match_percentage"] = 0

    # 6. Рандомизация результатов
    if randomness > 0 and len(candidates) > n_top:
        n_fixed = int(n_top * (1 - randomness))
        top_part = candidates.iloc[:n_fixed]
        random_part = candidates.iloc[n_fixed : n_top * 3].sample(n=n_top - n_fixed)
        final_df = pd.concat([top_part, random_part])
    else:
        final_df = candidates.head(n_top)

    # 7. Финальная выборка объектов из БД
    final_ids = final_df["Id_Recipe"].tolist() # Используем Id_Recipe
    rel_map = dict(zip(final_df["Id_Recipe"], final_df["match_percentage"]))
    abs_map = dict(zip(final_df["Id_Recipe"], final_df["abs_fridge_match"]))
    
    if not final_ids:
        return Recipe.objects.none()
        
    # Сохраняем порядок выдачи на основе Id_Recipe
    preserved_order = django.db.models.Case(
        *[django.db.models.When(Id_Recipe=rid, then=pos) for pos, rid in enumerate(final_ids)]
    )
    
    results = Recipe.objects.filter(Id_Recipe__in=final_ids).order_by(preserved_order)
    
    for r in results:
        r.match_score = rel_map.get(r.Id_Recipe, 0)
        r.fridge_match = abs_map.get(r.Id_Recipe, 0)
        
    return results

if __name__ == "__main__":
    try:
        TEST_USER_ID = 9 
        my_fridge = ["картофель", "лук", "курица", "соль", 'лимон', "чеснок", "Молоко", "корица"] 
        
        print(f"\nИСТОРИЯ ПОЛЬЗОВАТЕЛЯ ID {TEST_USER_ID}:")
        reactions = RecipeReaction.objects.filter(user_id=TEST_USER_ID)
        for r in reactions:
            print(f"{'👍' if r.reaction=='like' else '👎'} {r.recipe.Name_recipe}")

        print(f'\nХолодильник: {my_fridge}')
        print(f"\nРЕКОМЕНДАЦИИ:")
        recs = get_recommendations(
            TEST_USER_ID, 
            n_top=7, 
            fridge_ingredients=my_fridge,
            randomness=0.4
        )
        
        for r in recs:
            print(f"[{r.match_score}% Подходит] | "
                  f"[Наличие продуктов: {r.fridge_match}%] | "
                  f"ID_Recipe: {r.Id_Recipe} | {r.Name_recipe}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()