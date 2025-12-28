import os
import django
import sys
import pandas as pd
import ast
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- БЛОК ИНИЦИАЛИЗАЦИИ DJANGO ---

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

django.setup()

from myapp.models import Recipe, RecipeReaction

# --- 1. Вспомогательные функции ---

def safe_literal_eval(val):
    """Безопасно преобразует строку в объект Python."""
    if not val or not isinstance(val, str):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def parse_ingredients(ingredients_str):
    """Извлекает названия ингредиентов из JSON-подобной строки."""
    parsed_list = safe_literal_eval(ingredients_str)
    ingredients = []
    if isinstance(parsed_list, list):
        for item in parsed_list:
            if isinstance(item, list) and len(item) > 1:
                ingredients.append(item[1])
    return ingredients

# --- 2. Подготовка данных ---

def get_data_from_db():
    """Загружает данные из моделей Django."""
    recipes = Recipe.objects.all().values()
    recipes_df = pd.DataFrame(list(recipes))
    
    interactions = RecipeReaction.objects.all().values('user_id', 'recipe_id', 'reaction')
    interactions_df = pd.DataFrame(list(interactions))
    
    if not interactions_df.empty:
        interactions_df['interaction'] = interactions_df['reaction'].map({
            'like': 1.0,
            'dislike': -1.0
        }).fillna(0.0)
    
    return recipes_df, interactions_df

def preprocess_recipes(recipes_df):
    """Создание признаков для контентной фильтрации."""
    if recipes_df.empty:
        return recipes_df

    recipes_df['parsed_tags'] = recipes_df['Tags'].apply(safe_literal_eval)
    recipes_df['parsed_ingredients'] = recipes_df['Ingredients'].apply(parse_ingredients)
    
    recipes_df['Name_Desc'] = recipes_df['Name_recipe'].fillna('') + ' ' + recipes_df['Description'].fillna('')
    recipes_df['Tags_str'] = recipes_df['parsed_tags'].apply(lambda x: ' '.join(x))
    recipes_df['Ingredients_str'] = recipes_df['parsed_ingredients'].apply(lambda x: ' '.join(x))
    
    return recipes_df

# --- 3. Алгоритмы ---

def build_cb_matrix(recipes_df):
    tfidf = TfidfVectorizer(stop_words=None)
    sim_name = cosine_similarity(tfidf.fit_transform(recipes_df['Name_Desc']))
    sim_tags = cosine_similarity(tfidf.fit_transform(recipes_df['Tags_str']))
    sim_ingr = cosine_similarity(tfidf.fit_transform(recipes_df['Ingredients_str']))
    return (0.3 * sim_name) + (0.4 * sim_tags) + (0.3 * sim_ingr)

def get_popularity_scores(recipes_df):
    scores = np.log1p(recipes_df['Likes'].astype(float) + recipes_df['Safes'].astype(float))
    if scores.max() > scores.min():
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = scores * 0
    return pd.Series(normalized.values, index=recipes_df['id']).to_dict()

# --- 4. Основная функция с параметром рандома ---

def get_recommendations_for_user(sent_user_id, n_top=5, random_offset=0.2):
    """
    Параметры:
    - sent_user_id: ID пользователя
    - n_top: сколько рецептов вернуть
    - random_offset: доля случайных рецептов (0.2 = 20% новых открытий)
    """
    recipes_df_raw, interactions_df = get_data_from_db()
    
    if recipes_df_raw.empty:
        return Recipe.objects.none()

    recipes_df = preprocess_recipes(recipes_df_raw)
    item_id_to_idx = pd.Series(recipes_df.index, index=recipes_df['id']).to_dict()
    
    user_history = interactions_df[interactions_df['user_id'] == sent_user_id] if not interactions_df.empty else pd.DataFrame()
    liked_ids = user_history[user_history['interaction'] > 0]['recipe_id'].tolist()
    seen_ids = user_history['recipe_id'].tolist()

    # Расчет базовых скоров
    if not liked_ids:
        # Для новых — популярное
        popularity = get_popularity_scores(recipes_df)
        recipes_df['score'] = recipes_df['id'].map(popularity)
    else:
        # Для существующих — контентный анализ
        cb_matrix = build_cb_matrix(recipes_df)
        liked_indices = [item_id_to_idx[rid] for rid in liked_ids if rid in item_id_to_idx]
        
        if liked_indices:
            user_profile_sim = cb_matrix[liked_indices].mean(axis=0)
            recipes_df['score'] = user_profile_sim
        else:
            recipes_df['score'] = 0

    # Исключаем то, что пользователь уже видел или дизлайкнул
    candidates = recipes_df[~recipes_df['id'].isin(seen_ids)].copy()
    
    if candidates.empty:
        return Recipe.objects.none()

    # --- ЛОГИКА РАНДОМА ---
    # Берем чуть больше кандидатов, чем нужно (в 3 раза больше n_top)
    top_candidates = candidates.sort_values(by='score', ascending=False).head(n_top * 3)
    
    # Разделяем на "лучшие по алгоритму" и "случайные из новых"
    num_random = int(n_top * random_offset)
    num_fixed = n_top - num_random
    
    # 1. Берем фиксированную базу лучших соответствий
    final_ids = top_candidates.head(num_fixed)['id'].tolist()
    
    # 2. Добавляем случайные из оставшихся топ-кандидатов (новые, которые могут понравиться)
    remaining_pool = top_candidates[~top_candidates['id'].isin(final_ids)]
    if not remaining_pool.empty:
        random_ids = remaining_pool.sample(min(num_random, len(remaining_pool)))['id'].tolist()
        final_ids.extend(random_ids)
    
    # Перемешиваем итоговый список, чтобы случайные рецепты не всегда были в конце
    random.shuffle(final_ids)

    return Recipe.objects.filter(id__in=final_ids[:n_top])

# --- Запуск теста ---
if __name__ == "__main__":
    try:
        user_id_to_test = 9
        
        # Добавляем 20% случайных новых рецептов
        recs = get_recommendations_for_user(user_id_to_test, n_top=7, random_offset=0.2)
        
        print(f"--- Рекомендации для пользователя {user_id_to_test} ---")
        if recs.exists():
            for recipe in recs:
                print(f"Рецепт {recipe.Id_Recipe}: {recipe.Name_recipe}")
        else:
            print("Рецептов не найдено.")
            
    except Exception as e:
        print(f"Ошибка: {e}")