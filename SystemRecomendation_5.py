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

from myapp.models import Recipe, RecipeReaction, CustomUser

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
                ingredients.append(item[1].lower().strip())
    return ingredients

# --- 2. Подготовка данных ---

def get_data_from_db():
    """Загружает данные из моделей Django."""
    recipes = Recipe.objects.all().values()
    recipes_df = pd.DataFrame(list(recipes))
    
    interactions = RecipeReaction.objects.all().values('user_id', 'recipe_id', 'reaction')
    interactions_df = pd.DataFrame(list(interactions))
    
    if not interactions_df.empty:
        interactions_df['interaction_value'] = interactions_df['reaction'].map({
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
    """Строит матрицу сходства на основе ТФ-ИДФ."""
    tfidf = TfidfVectorizer(stop_words=None)
    # Важно: если данных мало, Tfidf может упасть, добавим проверку
    def get_sim(col):
        matrix = tfidf.fit_transform(recipes_df[col])
        return cosine_similarity(matrix)

    sim_name = get_sim('Name_Desc')
    sim_tags = get_sim('Tags_str')
    sim_ingr = get_sim('Ingredients_str')
    
    # Итоговая матрица сходства (Weighted)
    return (0.3 * sim_name) + (0.4 * sim_tags) + (0.3 * sim_ingr)

def get_popularity_scores(recipes_df):
    """Нормализованный лог-скор популярности."""
    scores = np.log1p(recipes_df['Likes'].astype(float) + recipes_df['Safes'].astype(float))
    if scores.max() > scores.min():
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = scores * 0
    return pd.Series(normalized.values, index=recipes_df['id']).to_dict()

# --- 4. Логика холодильника и Холодного старта ---

def calculate_fridge_boost(recipe_ingredients, user_fridge_list):
    """
    Рассчитывает коэффициент соответствия ингредиентам в наличии.
    Возвращает % ингредиентов рецепта, которые есть у пользователя.
    """
    if not user_fridge_list or not recipe_ingredients:
        return 1.0
    
    recipe_set = set(recipe_ingredients)
    user_set = set([i.lower().strip() for i in user_fridge_list])
    
    matches = recipe_set.intersection(user_set)
    # Возвращаем процент совпадения (от 0.5 до 1.5 как множитель)
    overlap_ratio = len(matches) / len(recipe_set) if len(recipe_set) > 0 else 0
    return 1.0 + overlap_ratio  # Тот, где больше совпадений, получит буст до х2

# --- 5. Основная функция ---

def get_recommendations_for_user(sent_user_id, n_top=5, random_offset=0.2, use_fridge=False, fridge_ingredients=None):
    """
    Параметры:
    - sent_user_id: ID пользователя
    - n_top: количество рецептов
    - random_offset: доля случайности
    - use_fridge: учитывать ли продукты в наличии
    - fridge_ingredients: список строк-ингредиентов пользователя
    """
    recipes_df_raw, interactions_df = get_data_from_db()
    if recipes_df_raw.empty:
        return Recipe.objects.none()

    recipes_df = preprocess_recipes(recipes_df_raw)
    item_id_to_idx = pd.Series(recipes_df.index, index=recipes_df['id']).to_dict()
    
    # Данные пользователя
    user_history = interactions_df[interactions_df['user_id'] == sent_user_id] if not interactions_df.empty else pd.DataFrame()
    liked_ids = user_history[user_history['interaction_value'] > 0]['recipe_id'].tolist()
    disliked_ids = user_history[user_history['interaction_value'] < 0]['recipe_id'].tolist()
    seen_ids = user_history['recipe_id'].tolist()

    # Поиск "предпочтений" из профиля (Cold Start по ингредиентам)
    user_obj = CustomUser.objects.filter(id=sent_user_id).first()
    # Предположим, в id_liked_recipes мы храним ID ингредиентов или названий для холодного старта
    # В данном примере используем fridge_ingredients как основу для интереса, если лайков нет

    # РАСЧЕТ БАЗОВОГО SCORE
    if not liked_ids and not (use_fridge and fridge_ingredients):
        # Совсем новый пользователь -> Популярное
        popularity = get_popularity_scores(recipes_df)
        recipes_df['score'] = recipes_df['id'].map(popularity)
    else:
        cb_matrix = build_cb_matrix(recipes_df)
        
        # 1. Учет Лайков и Дизлайков (Negative Sampling)
        # Формула: Profile = Avg(Likes) - 0.5 * Avg(Dislikes)
        liked_indices = [item_id_to_idx[rid] for rid in liked_ids if rid in item_id_to_idx]
        disliked_indices = [item_id_to_idx[rid] for rid in disliked_ids if rid in item_id_to_idx]
        
        # Вектор интереса
        user_profile_vector = np.zeros(len(recipes_df))
        
        if liked_indices:
            user_profile_vector += cb_matrix[liked_indices].mean(axis=0)
        
        if disliked_indices:
            # Вычитаем дизлайки с весом 0.5
            user_profile_vector -= 0.5 * cb_matrix[disliked_indices].mean(axis=0)
        
        # Если лайков нет, но есть ингредиенты "холодного старта" (используем fridge_ingredients как фильтр интереса)
        if not liked_indices and fridge_ingredients:
             # Создаем виртуальный скор на основе совпадения ингредиентов
             recipes_df['score'] = recipes_df['parsed_ingredients'].apply(
                 lambda x: calculate_fridge_boost(x, fridge_ingredients) - 1.0
             )
        else:
            recipes_df['score'] = user_profile_vector

    # 2. Фильтр по Холодильнику (Optional Boost)
    if use_fridge and fridge_ingredients:
        recipes_df['fridge_multiplier'] = recipes_df['parsed_ingredients'].apply(
            lambda x: calculate_fridge_boost(x, fridge_ingredients)
        )
        recipes_df['score'] = recipes_df['score'] * recipes_df['fridge_multiplier']

    # Исключаем увиденное
    candidates = recipes_df[~recipes_df['id'].isin(seen_ids)].copy()
    if candidates.empty:
        return Recipe.objects.none()

    # ЛОГИКА РАНДОМА И ВЫДАЧИ
    top_candidates = candidates.sort_values(by='score', ascending=False).head(n_top * 3)
    num_random = int(n_top * random_offset)
    num_fixed = n_top - num_random
    
    final_ids = top_candidates.head(num_fixed)['id'].tolist()
    remaining_pool = top_candidates[~top_candidates['id'].isin(final_ids)]
    
    if not remaining_pool.empty:
        random_ids = remaining_pool.sample(min(num_random, len(remaining_pool)))['id'].tolist()
        final_ids.extend(random_ids)
    
    random.shuffle(final_ids)
    return Recipe.objects.filter(id__in=final_ids[:n_top])

# --- БЛОК ТЕСТИРОВАНИЯ ---
if __name__ == "__main__":
    try:
        # Имитируем ID пользователя (убедитесь, что он есть в БД)
        TEST_USER_ID = 9 
        
        # Пример 1: Рекомендации с учетом холодильника (Холодный старт)
        my_fridge = ["картофель"]
        
        print(f"\n--- Рекомендации для пользователя {TEST_USER_ID} (С холодильником) ---")
        recs_fridge = get_recommendations_for_user(
            TEST_USER_ID, 
            n_top=5, 
            use_fridge=True, 
            fridge_ingredients=my_fridge
        )
        
        for r in recs_fridge:
            print(f"ID: {r.id} | {r.Name_recipe}")

        # Пример 2: Обычные рекомендации (учет дизлайков)
        print(f"\n--- Рекомендации для пользователя {TEST_USER_ID} (Обычный режим) ---")
        recs_standard = get_recommendations_for_user(TEST_USER_ID, n_top=5, use_fridge=False)
        
        for r in recs_standard:
            print(f"ID: {r.id} | {r.Name_recipe}")
            
    except Exception as e:
        import traceback
        print(f"Ошибка: {e}")
        traceback.print_exc()