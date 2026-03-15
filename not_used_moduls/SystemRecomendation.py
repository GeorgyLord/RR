import pandas as pd
import io
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import sqlite3 

# --- 1. Функции для обработки данных ---

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
    Пример: "[['0', 'Говяжьи кости', '2 кг'], ...]" -> ['Говяжьи кости', ...]
    """
    parsed_list = safe_literal_eval(ingredients_str)
    ingredients = []
    if isinstance(parsed_list, list):
        for item in parsed_list:
            if isinstance(item, list) and len(item) > 1:
                ingredients.append(item[1])  # Индекс 1 - название ингредиента
    return ingredients

# --- 2. Функция: Загрузка данных из SQLite ---

def load_data_from_sqlite(db_filepath, recipes_table_name, interactions_table_name):
    """
    Загружает данные из таблиц SQLite.
    """
    print(f"Подключение к базе данных: {db_filepath}")
    try:
        conn = sqlite3.connect(db_filepath)
        
        # Чтение таблицы рецептов
        query_recipes = f"SELECT * FROM {recipes_table_name}"
        recipes_df = pd.read_sql_query(query_recipes, conn)
        print(f"Загружено {len(recipes_df)} рецептов из таблицы '{recipes_table_name}'.")

        # Чтение таблицы взаимодействий
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

# --- 3. Функция: Подготовка данных ---

def load_and_preprocess_data(recipes_df, interactions_df):
    """
    Подготавливает данные (парсинг и создание признаков).
    """
    
    if recipes_df.empty or interactions_df.empty:
        print("Предупреждение: Пустые данные для обработки.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Подготовка recipes_df для Content-Based ---
    recipes_df = recipes_df.fillna('')

    # Парсинг тегов и ингредиентов (с проверкой столбцов)
    if 'Tags' in recipes_df.columns:
        recipes_df['parsed_tags'] = recipes_df['Tags'].apply(safe_literal_eval)
    else:
        recipes_df['parsed_tags'] = [[]] * len(recipes_df)

    if 'Ingredients' in recipes_df.columns:
        recipes_df['parsed_ingredients'] = recipes_df['Ingredients'].apply(parse_ingredients)
    else:
        recipes_df['parsed_ingredients'] = [[]] * len(recipes_df)

    # Создание отдельных признаков для Feature Fusion
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

# --- 4. Модели Content-Based (CB) и Collaborative Filtering (CF) ---

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
    
    # 1. Создание матрицы "пользователь-рецепт"
    user_item_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='interaction'
    ).fillna(0)
    
    # 2. Создание матрицы схожести пользователей
    user_similarity = cosine_similarity(user_item_matrix)
    
    # 3. Преобразование в DataFrame для удобного поиска по user_id
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


def get_hybrid_recommendations(user_id, recipes_df, interactions_df, 
                               cb_cosine_sim, item_id_to_index_map, 
                               cf_user_item_matrix, cf_user_sim_df,
                               n=5, alpha=0.5):
    """
    Генерирует гибридные рекомендации для пользователя.
    """
    print(f"\n--- Генерация гибридных рекомендаций для user_id={user_id} ---")
    print(f"Веса: CF(kNN)={alpha*100}%, CB(Feature Fusion)={(1-alpha)*100}%")

    # 1. Получаем список "понравившихся" и "уже просмотренных"
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_items = user_interactions[user_interactions['interaction'] == 1]['item_id'].tolist()
    seen_items = user_interactions['item_id'].unique().tolist()

    # 2. Получаем список всех item_id, которые пользователь ЕЩЕ НЕ видел
    all_item_ids = recipes_df['id'].unique()
    items_to_score = [item_id for item_id in all_item_ids if item_id not in seen_items]

    if not items_to_score:
        print("Пользователь оценил все доступные рецепты.")
        return pd.DataFrame(columns=['id', 'Name_recipe', 'hybrid_score'])

    # 3. Расчет "профиля" пользователя для CB
    if liked_items:
        liked_indices = [item_id_to_index_map[item_id] for item_id in liked_items if item_id in item_id_to_index_map]
        if liked_indices:
            user_cb_profile = cb_cosine_sim[liked_indices].mean(axis=0)
        else:
            user_cb_profile = np.zeros(len(item_id_to_index_map))
    else:
        user_cb_profile = np.zeros(len(item_id_to_index_map))

    # 4. Расчет гибридной оценки для каждого не просмотренного рецепта
    recommendations = []
    
    for item_id in items_to_score:
        if item_id not in item_id_to_index_map:
            continue

        item_idx = item_id_to_index_map[item_id]
        
        # --- CF Оценка (k-NN) ---
        cf_score = predict_cf_knn(
            user_id, item_id, 
            cf_user_item_matrix, cf_user_sim_df, 
            k=5
        )
        # Нормализация CF оценки к [0, 1]
        cf_score_normalized = (cf_score + 1) / 2.0
        
        # --- CB Оценка ---
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


# --- 5. Исправленный основной блок выполнения (start_fun) ---

@lru_cache(maxsize=128)
def start_fun(n, top=5, _DB_FILE_PATH = "db.sqlite3", _RECIPES_TABLE = "myapp_recipe", _INTERACTIONS_TABLE = "myapp_recipereaction"):
    """
    Основная функция для запуска системы рекомендаций с загрузкой данных из SQLite.
    
    Установлены значения по умолчанию, соответствующие вашей структуре.
    """
    
    DB_FILE_PATH = _DB_FILE_PATH
    RECIPES_TABLE = _RECIPES_TABLE
    INTERACTIONS_TABLE = _INTERACTIONS_TABLE
    
    # 1. Загрузка данных из SQLite
    recipes_df_raw, interactions_df_raw = load_data_from_sqlite(DB_FILE_PATH, RECIPES_TABLE, INTERACTIONS_TABLE)
    
    if recipes_df_raw.empty or interactions_df_raw.empty:
        print("Данные из базы данных не загружены. Выход.")
        return pd.DataFrame()

    # --- ИСПРАВЛЕНИЕ: ПЕРЕИМЕНОВАНИЕ И ПРЕОБРАЗОВАНИЕ КОЛОНОК СОГЛАСНО СХЕМЕ ---

    # 1. Переименование столбца рецепта: recipe_id (Ваша БД) -> item_id (Модель)
    if 'recipe_id' in interactions_df_raw.columns and 'item_id' not in interactions_df_raw.columns:
        interactions_df_raw = interactions_df_raw.rename(columns={'recipe_id': 'item_id'})
        print("Переименован столбец 'recipe_id' в 'item_id' для CF.")
    
    # 2. Переименование столбца оценки: reaction (Ваша БД) -> interaction (Модель)
    if 'reaction' in interactions_df_raw.columns and 'interaction' not in interactions_df_raw.columns:
        interactions_df_raw = interactions_df_raw.rename(columns={'reaction': 'interaction'})
        print("Переименован столбец 'reaction' в 'interaction' для CF.")
    
    # 3. Преобразование строковых оценок в числовой формат (1.0/-1.0)
    # Если столбец 'interaction' содержит строки ('like', 'dislike'), преобразуем их.
    if 'interaction' in interactions_df_raw.columns and interactions_df_raw['interaction'].dtype == object:
        # Считаем 'like' как 1.0, 'dislike' как -1.0, остальные как 0.0
        interactions_df_raw['interaction'] = interactions_df_raw['interaction'].astype(str).str.lower().map({
            'like': 1.0,
            'dislike': -1.0,
            'none': 0.0,
        }).fillna(0.0)
        print("Столбец 'interaction' (реакция) преобразован в числовой формат (1.0/-1.0).")

    # Проверка на наличие необходимых столбцов после переименования
    required_cols = ['user_id', 'item_id', 'interaction']
    if not all(col in interactions_df_raw.columns for col in required_cols):
        print(f"ОШИБКА: Отсутствуют необходимые столбцы в interactions_df. Требуются: {required_cols}")
        return pd.DataFrame()
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ПЕРЕИМЕНОВАНИЯ И ПРЕОБРАЗОВАНИЯ ---
        
    # 2. Подготовка данных (парсинг и создание признаков)
    recipes_df, interactions_df = load_and_preprocess_data(recipes_df_raw, interactions_df_raw)
    
    # Обеспечение числового типа данных для id и item_id
    if 'id' in recipes_df.columns:
        recipes_df['id'] = pd.to_numeric(recipes_df['id'], errors='coerce')
        recipes_df = recipes_df.dropna(subset=['id'])
        recipes_df['id'] = recipes_df['id'].astype(int)
    
    if 'item_id' in interactions_df.columns:
        interactions_df['item_id'] = pd.to_numeric(interactions_df['item_id'], errors='coerce')
        interactions_df = interactions_df.dropna(subset=['item_id'])
        interactions_df['item_id'] = interactions_df['item_id'].astype(int)

    # 3. Создание карты {item_id: index}
    item_id_to_index_map = pd.Series(recipes_df.index, index=recipes_df['id']).to_dict()

    # 4. Обучение моделей
    # Модель CF (kNN)
    cf_user_item_matrix, cf_user_sim_df = build_cf_components(interactions_df)
    # Модель CB (Feature Fusion)
    cb_cosine_sim_matrix = build_cb_model(recipes_df)

    # --- Демонстрация ---
    
    TEST_USER_ID = n
    
    # Проверка существования пользователя перед поиском взаимодействий
    if TEST_USER_ID not in interactions_df['user_id'].unique():
         print(f"Тестовый пользователь с ID {TEST_USER_ID} не найден в данных взаимодействий.")
         return pd.DataFrame()
         
    user_likes = interactions_df[
        (interactions_df['user_id'] == TEST_USER_ID) & 
        (interactions_df['interaction'] == 1)
    ]['item_id'].tolist()
        
    liked_recipes = recipes_df[recipes_df['id'].isin(user_likes)]['Name_recipe'].tolist()
    print(f"\nПользователю {TEST_USER_ID} понравились: {liked_recipes}")

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
        n=top,
        alpha=RECOMMENDED_ALPHA
    )
    
    return hybrid_recs_03


# Пример вызова функции:
print(start_fun(9, 5))