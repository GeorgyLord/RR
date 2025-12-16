import pandas as pd
import io
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import sqlite3 

# --- 1. Функции для обработки данных (без изменений) ---

def safe_literal_eval(val):
    """
    Безопасно парсит строку как Python-объект (например, список).
    Возвращает пустой список в случае ошибки.
    """
    if not isinstance(val, str):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

def parse_ingredients(ingredients_str):
    """
    Извлекает только названия ингредиентов из сложной строки.
    """
    parsed_list = safe_literal_eval(ingredients_str)
    ingredients = []
    if isinstance(parsed_list, list):
        for item in parsed_list:
            if isinstance(item, list) and len(item) > 1:
                ingredients.append(item[1])  # Индекс 1 - название ингредиента
    return ingredients

# --- 2. Функция: Загрузка данных из SQLite (без изменений) ---

def load_data_from_sqlite(db_filepath, recipes_table_name, interactions_table_name):
    """
    Загружает данные из таблиц SQLite.
    """
    print(f"Подключение к базе данных: {db_filepath}")
    try:
        conn = sqlite3.connect(db_filepath)
        
        query_recipes = f"SELECT * FROM {recipes_table_name}"
        recipes_df = pd.read_sql_query(query_recipes, conn)
        print(f"Загружено {len(recipes_df)} рецептов из таблицы '{recipes_table_name}'.")

        query_interactions = f"SELECT * FROM {interactions_table_name}"
        interactions_df = pd.read_sql_query(query_interactions, conn)
        print(f"Загружено {len(interactions_df)} взаимодействий из таблицы '{interactions_table_name}'.")

        conn.close()
        return recipes_df, interactions_df
    
    except sqlite3.Error as e:
        print(f"ОШИБКА при работе с SQLite: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке данных: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. Функция: Подготовка данных (без изменений) ---

def load_and_preprocess_data(recipes_df, interactions_df):
    """
    Подготавливает данные (парсинг и создание признаков).
    """
    
    if recipes_df.empty or interactions_df.empty:
        print("Предупреждение: Пустые данные для обработки.")
        return pd.DataFrame(), pd.DataFrame()

    recipes_df = recipes_df.fillna('')

    if 'Tags' in recipes_df.columns:
        recipes_df['parsed_tags'] = recipes_df['Tags'].apply(safe_literal_eval)
    else:
        recipes_df['parsed_tags'] = [[]] * len(recipes_df)

    if 'Ingredients' in recipes_df.columns:
        recipes_df['parsed_ingredients'] = recipes_df['Ingredients'].apply(parse_ingredients)
    else:
        recipes_df['parsed_ingredients'] = [[]] * len(recipes_df)

    name_col = 'Name_recipe' if 'Name_recipe' in recipes_df.columns else 'Name' 
    desc_col = 'Description' if 'Description' in recipes_df.columns else 'Description'
    
    recipes_df['Name_Desc'] = recipes_df.apply(
        lambda row: (
            str(row[name_col]) + ' ' +
            str(row[desc_col])
        ),
        axis=1
    )
    recipes_df['Tags_str'] = recipes_df['parsed_tags'].apply(lambda x: ' '.join(x))
    recipes_df['Ingredients_str'] = recipes_df['parsed_ingredients'].apply(lambda x: ' '.join(x))
    
    recipes_df = recipes_df.reset_index(drop=True)

    return recipes_df, interactions_df

# --- 4. Модели Content-Based (CB) и Collaborative Filtering (CF) (без изменений) ---

def build_cb_model(recipes_df):
    """
    Создает TF-IDF векторизаторы для нескольких признаков и объединяет матрицы сходства с весами.
    """
    print("Обучение Content-Based модели (Feature Fusion с TF-IDF)...")
    
    WEIGHT_NAME_DESC = 0.3
    WEIGHT_TAGS = 0.4
    WEIGHT_INGREDIENTS = 0.3
    
    tfidf = TfidfVectorizer(stop_words=None)
    
    tfidf_name_desc = tfidf.fit_transform(recipes_df['Name_Desc'])
    sim_name_desc = cosine_similarity(tfidf_name_desc, tfidf_name_desc)
    print(" - Сходство по Описанию рассчитано.")

    tfidf_tags = tfidf.fit_transform(recipes_df['Tags_str'])
    sim_tags = cosine_similarity(tfidf_tags, tfidf_tags)
    print(" - Сходство по Тегам рассчитано.")

    tfidf_ingredients = tfidf.fit_transform(recipes_df['Ingredients_str'])
    sim_ingredients = cosine_similarity(tfidf_ingredients, tfidf_ingredients)
    print(" - Сходство по Ингредиентам рассчитано.")
    
    weighted_sim_matrix = (
        (WEIGHT_NAME_DESC * sim_name_desc) +
        (WEIGHT_TAGS * sim_tags) +
        (WEIGHT_INGREDIENTS * sim_ingredients)
    )
    
    print(f"Матрица сходства (CB, Fusion) создана с весами: Name/Desc={WEIGHT_NAME_DESC}, Tags={WEIGHT_TAGS}, Ingredients={WEIGHT_INGREDIENTS}.")
    return weighted_sim_matrix


def build_cf_components(interactions_df):
    """
    Создает компоненты для User-Based k-NN CF.
    """
    print("Создание компонентов Collaborative Filtering (User-kNN)...")
    
    user_item_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='interaction'
    ).fillna(0)
    
    user_similarity = cosine_similarity(user_item_matrix)
    
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    print("Компоненты CF (User-kNN) созданы.")
    return user_item_matrix, user_similarity_df

def predict_cf_knn(user_id, item_id, user_item_matrix, user_similarity_df, k=5):
    """
    Предсказывает оценку пользователя для рецепта, используя User-Based k-NN.
    """
    if user_id not in user_similarity_df.index:
        return 0.0 
    
    user_sims = user_similarity_df[user_id]
    
    try:
        item_ratings = user_item_matrix[item_id]
    except KeyError:
        return 0.0 

    raters_indices = item_ratings[item_ratings != 0].index
    
    if len(raters_indices) == 0:
        return 0.0
        
    sims_of_raters = user_sims[raters_indices]
    ratings_of_raters = item_ratings[raters_indices]

    sims_of_raters = sims_of_raters[sims_of_raters > 0]
    if sims_of_raters.empty:
        return 0.0
        
    top_k_similar_users = sims_of_raters.sort_values(ascending=False).head(k)
    top_k_ratings = ratings_of_raters[top_k_similar_users.index]
    
    weighted_sum = np.dot(top_k_ratings, top_k_similar_users)
    sum_of_weights = top_k_similar_users.sum()
    
    if sum_of_weights == 0:
        return 0.0
        
    predicted_rating = weighted_sum / sum_of_weights
    
    return np.clip(predicted_rating, -1, 1)

# --- 5. НОВАЯ ФУНКЦИЯ: Расчет популярности (для Cold Start) ---

def get_recipe_popularity_scores(recipes_df):
    """
    Рассчитывает и нормализует оценку популярности рецептов на основе Likes и Safes.
    Возвращает словарь {item_id: normalized_popularity_score}.
    """
    print("Расчет популярности рецептов...")
    
    # Убедимся, что столбцы существуют и имеют числовой тип
    if 'Likes' in recipes_df.columns and 'Safes' in recipes_df.columns:
        recipes_df['Likes'] = pd.to_numeric(recipes_df['Likes'], errors='coerce').fillna(0)
        recipes_df['Safes'] = pd.to_numeric(recipes_df['Safes'], errors='coerce').fillna(0)
        
        # Общая популярность: сумма лайков и сохранений (log-трансформация для сглаживания)
        recipes_df['Popularity_Score'] = np.log1p(recipes_df['Likes'] + recipes_df['Safes'])
    else:
        print("Внимание: Столбцы 'Likes' или 'Safes' отсутствуют. Популярность установлена в 0.")
        recipes_df['Popularity_Score'] = 0.0
    
    # Нормализация оценки к диапазону [0, 1]
    min_score = recipes_df['Popularity_Score'].min()
    max_score = recipes_df['Popularity_Score'].max()
    
    if max_score > min_score:
        recipes_df['Popularity_Score_Norm'] = (recipes_df['Popularity_Score'] - min_score) / (max_score - min_score)
    else:
        recipes_df['Popularity_Score_Norm'] = 0.0

    # Создаем словарь {item_id: normalized_popularity_score}
    popularity_scores = pd.Series(recipes_df['Popularity_Score_Norm'].values, index=recipes_df['id']).to_dict()
    print("Расчет популярности завершен.")
    return popularity_scores


# --- 6. МОДИФИЦИРОВАННАЯ ГИБРИДНАЯ ФУНКЦИЯ ---

def get_hybrid_recommendations(user_id, recipes_df, interactions_df, 
                               cb_cosine_sim, item_id_to_index_map, 
                               cf_user_item_matrix, cf_user_sim_df,
                               popularity_scores, # НОВЫЙ АРГУМЕНТ
                               n=5, alpha=0.5):
    """
    Генерирует гибридные рекомендации для пользователя.
    В случае "холодного старта" использует популярность.
    """
    print(f"\n--- Генерация гибридных рекомендаций для user_id={user_id} ---")

    # 1. Определяем "понравившиеся" рецепты
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_items = user_interactions[user_interactions['interaction'] == 1]['item_id'].tolist()
    seen_items = user_interactions['item_id'].unique().tolist()
    
    # ОПРЕДЕЛЕНИЕ ХОЛОДНОГО СТАРТА
    is_cold_start = not liked_items
    
    if is_cold_start:
        print(f"Пользователь {user_id} - Холодный старт (нет лайков). Используется Popularity-Based CB.")
    else:
        print(f"Пользователь {user_id} - Теплый старт. Используется Hybrid CF + CB.")


    # 2. Получаем список всех item_id, которые пользователь ЕЩЕ НЕ видел
    all_item_ids = recipes_df['id'].unique()
    items_to_score = [item_id for item_id in all_item_ids if item_id not in seen_items]

    if not items_to_score:
        print("Пользователь оценил все доступные рецепты.")
        return pd.DataFrame(columns=['id', 'Name_recipe', 'hybrid_score'])

    # 3. Расчет "профиля" пользователя для CB (только если это НЕ холодный старт)
    if not is_cold_start:
        liked_indices = [item_id_to_index_map[item_id] for item_id in liked_items if item_id in item_id_to_index_map]
        if liked_indices:
            # Усреднение профиля пользователя по его понравившимся рецептам
            user_cb_profile = cb_cosine_sim[liked_indices].mean(axis=0)
        else:
            # По идее, этот блок не должен достигаться, если liked_items не пуст, но для безопасности:
            user_cb_profile = np.zeros(len(item_id_to_index_map))
    else:
        # В режиме холодного старта, CB профиль не используется
        user_cb_profile = np.zeros(len(item_id_to_index_map))

    # 4. Расчет гибридной оценки для каждого не просмотренного рецепта
    recommendations = []
    
    for item_id in items_to_score:
        if item_id not in item_id_to_index_map:
            continue

        item_idx = item_id_to_index_map[item_id]
        
        # --- CF Оценка (k-NN) ---
        # CF всегда предсказывает 0 для новых пользователей, что корректно.
        cf_score = predict_cf_knn(
            user_id, item_id, 
            cf_user_item_matrix, cf_user_sim_df, 
            k=5
        )
        # Нормализация CF оценки к [0, 1]
        cf_score_normalized = (cf_score + 1) / 2.0
        
        # --- CB Оценка / Популярность ---
        if is_cold_start:
            # Для холодного старта CB score - это популярность
            cb_score = popularity_scores.get(item_id, 0.0) 
        else:
            # Для "теплого" пользователя - обычный CB score
            cb_score = user_cb_profile[item_idx]
            
        # --- Гибридная Оценка ---
        hybrid_score = (alpha * cf_score_normalized) + ((1 - alpha) * cb_score)
        
        recommendations.append((item_id, hybrid_score, cf_score_normalized, cb_score))

    # 5. Сортировка и возврат топ-N
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    top_n_ids = [rec[0] for rec in recommendations[:n]]
    top_n_scores = [rec[1] for rec in recommendations[:n]]
    top_n_cf_scores = [rec[2] for rec in recommendations[:n]]
    top_n_cb_scores = [rec[3] for rec in recommendations[:n]]

    results_df = recipes_df[recipes_df['id'].isin(top_n_ids)].copy()
    results_df = results_df.set_index('id').loc[top_n_ids].reset_index() # Сохраняем порядок
    
    results_df['hybrid_score'] = top_n_scores
    results_df['cf_score_norm'] = top_n_cf_scores
    results_df['cb_score'] = top_n_cb_scores
    
    return results_df[['id', 'Name_recipe', 'hybrid_score', 'cf_score_norm', 'cb_score']]


# --- 7. МОДИФИЦИРОВАННЫЙ ОСНОВНОЙ БЛОК (start_fun) ---

@lru_cache(maxsize=128)
def start_fun(n, top=5, _DB_FILE_PATH = "db.sqlite3", _RECIPES_TABLE = "myapp_recipe", _INTERACTIONS_TABLE = "myapp_recipereaction"):
    
    DB_FILE_PATH = _DB_FILE_PATH
    RECIPES_TABLE = _RECIPES_TABLE
    INTERACTIONS_TABLE = _INTERACTIONS_TABLE
    
    # 1. Загрузка данных из SQLite
    recipes_df_raw, interactions_df_raw = load_data_from_sqlite(DB_FILE_PATH, RECIPES_TABLE, INTERACTIONS_TABLE)
    
    if recipes_df_raw.empty or interactions_df_raw.empty:
        print("Данные из базы данных не загружены. Выход.")
        return pd.DataFrame()

    # 2. ИСПРАВЛЕНИЕ: ПЕРЕИМЕНОВАНИЕ И ПРЕОБРАЗОВАНИЕ КОЛОНОК
    if 'recipe_id' in interactions_df_raw.columns and 'item_id' not in interactions_df_raw.columns:
        interactions_df_raw = interactions_df_raw.rename(columns={'recipe_id': 'item_id'})
        print("Переименован столбец 'recipe_id' в 'item_id' для CF.")
    
    if 'reaction' in interactions_df_raw.columns and 'interaction' not in interactions_df_raw.columns:
        interactions_df_raw = interactions_df_raw.rename(columns={'reaction': 'interaction'})
        print("Переименован столбец 'reaction' в 'interaction' для CF.")
    
    if 'interaction' in interactions_df_raw.columns and interactions_df_raw['interaction'].dtype == object:
        interactions_df_raw['interaction'] = interactions_df_raw['interaction'].astype(str).str.lower().map({
            'like': 1.0,
            'dislike': -1.0,
            'none': 0.0,
        }).fillna(0.0)
        print("Столбец 'interaction' (реакция) преобразован в числовой формат (1.0/-1.0).")

    required_cols = ['user_id', 'item_id', 'interaction']
    if not all(col in interactions_df_raw.columns for col in required_cols):
        print(f"ОШИБКА: Отсутствуют необходимые столбцы в interactions_df. Требуются: {required_cols}")
        return pd.DataFrame()
        
    # 3. Подготовка данных (парсинг и создание признаков)
    recipes_df, interactions_df = load_and_preprocess_data(recipes_df_raw, interactions_df_raw)
    
    # Обеспечение числового типа данных
    if 'id' in recipes_df.columns:
        recipes_df['id'] = pd.to_numeric(recipes_df['id'], errors='coerce')
        recipes_df = recipes_df.dropna(subset=['id'])
        recipes_df['id'] = recipes_df['id'].astype(int)
    
    if 'item_id' in interactions_df.columns:
        interactions_df['item_id'] = pd.to_numeric(interactions_df['item_id'], errors='coerce')
        interactions_df = interactions_df.dropna(subset=['item_id'])
        interactions_df['item_id'] = interactions_df['item_id'].astype(int)

    # 4. Создание карты {item_id: index}
    item_id_to_index_map = pd.Series(recipes_df.index, index=recipes_df['id']).to_dict()

    # 5. Обучение моделей
    # Модель CF (kNN)
    cf_user_item_matrix, cf_user_sim_df = build_cf_components(interactions_df)
    # Модель CB (Feature Fusion)
    cb_cosine_sim_matrix = build_cb_model(recipes_df)
    
    # НОВЫЙ ШАГ: Расчет популярности (для cold start)
    popularity_scores = get_recipe_popularity_scores(recipes_df) 

    # --- Демонстрация ---
    
    TEST_USER_ID = n
    
    # Проверка существования пользователя (позволяем идти дальше, если его нет)
    
    user_likes = interactions_df[
        (interactions_df['user_id'] == TEST_USER_ID) & 
        (interactions_df['interaction'] == 1)
    ]['item_id'].tolist()
        
    liked_recipes = recipes_df[recipes_df['id'].isin(user_likes)]['Name_recipe'].tolist()
    print(f"\nПользователю {TEST_USER_ID} понравились: {liked_recipes if liked_recipes else 'пока ничего (холодный старт)'}")

    # Получаем рекомендации с рекомендованным коэффициентом alpha = 0.3
    RECOMMENDED_ALPHA = 0.3
    
    hybrid_recs_03 = get_hybrid_recommendations(
        TEST_USER_ID, 
        recipes_df, 
        interactions_df, 
        cb_cosine_sim_matrix, 
        item_id_to_index_map,
        cf_user_item_matrix,
        cf_user_sim_df,
        popularity_scores, # Передача оценок популярности
        n=top,
        alpha=RECOMMENDED_ALPHA
    )
    
    return hybrid_recs_03


# Пример вызова функции:
# Если пользователь 8 имеет лайки, он получит персонализированные рекомендации.
# Если вы передадите ID, которого нет в базе, он получит популярные рецепты.
# print(start_fun(9, 5))