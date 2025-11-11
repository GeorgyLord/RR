import pandas as pd
import io
import ast  # Для безопасного парсинга строковых представлений списков
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# --- Импорты Surprise (Dataset, Reader, SVD, accuracy) удалены ---

# --- 1. Моделирование данных (вместо чтения из CSV-файлов) ---
# В реальном приложении вы будете использовать pd.read_csv('recipes.csv') и pd.read_csv('interactions.csv')

# --- УДАЛЕНЫ БОЛЬШИЕ СТРОКИ RECIPES_CSV_DATA и INTERACTIONS_CSV_DATA ---
# Теперь мы будем загружать данные из файлов в блоке if __name__ == "__main__"

# --- 2. Функции для обработки данных ---

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

def load_and_preprocess_data(recipes_filepath, interactions_filepath):
    """
    Загружает и подготавливает данные из ФАЙЛОВ.
    
    :param recipes_filepath: Строковый путь к CSV-файлу с рецептами
    :param interactions_filepath: Строковый путь к CSV-файлу с взаимодействиями
    """
    # Загрузка данных
    # УБРАН io.StringIO, теперь читаем напрямую из путей к файлам
    try:
        recipes_df = pd.read_csv(recipes_filepath)
        interactions_df = pd.read_csv(interactions_filepath)
    except FileNotFoundError as e:
        print(f"ОШИБКА: Файл не найден. Убедитесь, что пути верны. {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"ОШИБКА при чтении CSV: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # --- Подготовка recipes_df для Content-Based ---
    # Заполнение пропусков (например, в 'Description')
    recipes_df = recipes_df.fillna('')

    # Парсинг тегов и ингредиентов
    recipes_df['parsed_tags'] = recipes_df['Tags'].apply(safe_literal_eval)
    recipes_df['parsed_ingredients'] = recipes_df['Ingredients'].apply(parse_ingredients)

    # Создание отдельных признаков для Feature Fusion
    recipes_df['Name_Desc'] = recipes_df.apply(
        lambda row: (
            str(row['Name_recipe']) + ' ' +
            str(row['Description'])
        ),
        axis=1
    )
    recipes_df['Tags_str'] = recipes_df['parsed_tags'].apply(lambda x: ' '.join(x))
    recipes_df['Ingredients_str'] = recipes_df['parsed_ingredients'].apply(lambda x: ' '.join(x))
    
    # Сброс индекса, чтобы iloc (индекс по позиции) соответствовал
    recipes_df = recipes_df.reset_index(drop=True)

    return recipes_df, interactions_df

# --- 3. Модель Content-Based (CB) ---

def build_cb_model(recipes_df):
    """
    Создает TF-IDF векторизаторы для нескольких признаков и объединяет матрицы сходства с весами.
    Это повышает точность Content-Based модели, фокусируясь на ключевых атрибутах.
    """
    print("Обучение Content-Based модели (Feature Fusion с TF-IDF)...")
    
    # Конфигурация весов (для повышения точности CB)
    WEIGHT_NAME_DESC = 0.3
    WEIGHT_TAGS = 0.4
    WEIGHT_INGREDIENTS = 0.3
    
    tfidf = TfidfVectorizer(stop_words=None)
    
    # 1. Сходство по Названию и Описанию (Name_Desc)
    tfidf_name_desc = tfidf.fit_transform(recipes_df['Name_Desc'])
    sim_name_desc = cosine_similarity(tfidf_name_desc, tfidf_name_desc)
    print(" - Сходство по Описанию рассчитано.")

    # 2. Сходство по Тегам (Tags)
    tfidf_tags = tfidf.fit_transform(recipes_df['Tags_str'])
    sim_tags = cosine_similarity(tfidf_tags, tfidf_tags)
    print(" - Сходство по Тегам рассчитано.")

    # 3. Сходство по Ингредиентам (Ingredients)
    tfidf_ingredients = tfidf.fit_transform(recipes_df['Ingredients_str'])
    sim_ingredients = cosine_similarity(tfidf_ingredients, tfidf_ingredients)
    print(" - Сходство по Ингредиентам рассчитано.")
    
    # 4. Взвешенное объединение матриц сходства (Feature Fusion)
    weighted_sim_matrix = (
        (WEIGHT_NAME_DESC * sim_name_desc) +
        (WEIGHT_TAGS * sim_tags) +
        (WEIGHT_INGREDIENTS * sim_ingredients)
    )
    
    print(f"Матрица сходства (CB, Fusion) создана с весами: Name/Desc={WEIGHT_NAME_DESC}, Tags={WEIGHT_TAGS}, Ingredients={WEIGHT_INGREDIENTS}.")
    return weighted_sim_matrix

# --- 4. Модель Collaborative Filtering (CF) - БЕЗ SURPRISE ---
# ... Код CF остается без изменений ...

def build_cf_components(interactions_df):
    """
    Создает компоненты для User-Based k-NN CF.
    Возвращает матрицу "пользователь-рецепт" и матрицу схожести пользователей.
    """
    print("Создание компонентов Collaborative Filtering (User-kNN)...")
    
    # 1. Создание матрицы "пользователь-рецепт"
    # user_id - строки, item_id - столбцы, interaction - значения
    user_item_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='interaction'
    ).fillna(0)
    
    # 2. Создание матрицы схожести пользователей
    # Мы используем косинусное сходство между векторами оценок пользователей
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
    # 1. Проверка на "холодный старт" (новый пользователь)
    if user_id not in user_similarity_df.index:
        return 0.0 # Не можем ничего предсказать
    
    # 2. Получение вектора схожести этого пользователя с другими
    user_sims = user_similarity_df[user_id]
    
    # 3. Получение всех оценок для данного рецепта
    try:
        item_ratings = user_item_matrix[item_id]
    except KeyError:
        return 0.0 # "Холодный старт" для рецепта, его никто не оценил

    # 4. Находим пользователей, которые оценили этот рецепт (оценка != 0)
    raters_indices = item_ratings[item_ratings != 0].index
    
    # 5. Если никто не оценил, не можем предсказать
    if len(raters_indices) == 0:
        return 0.0
        
    # 6. Фильтруем схожести, оставляя только тех, кто оценил рецепт
    sims_of_raters = user_sims[raters_indices]
    
    # 7. Фильтруем оценки, оставляя только тех, кто оценил
    ratings_of_raters = item_ratings[raters_indices]

    # 8. Находим k-наиболее похожих пользователей
    # Игнорируем пользователей с отрицательной или нулевой схожестью
    sims_of_raters = sims_of_raters[sims_of_raters > 0]
    if sims_of_raters.empty:
        return 0.0 # Нет похожих пользователей, оценивших этот рецепт
        
    # 9. Сортируем и берем топ-k
    top_k_similar_users = sims_of_raters.sort_values(ascending=False).head(k)
    top_k_ratings = ratings_of_raters[top_k_similar_users.index]
    
    # 10. Расчет взвешенного среднего
    weighted_sum = np.dot(top_k_ratings, top_k_similar_users)
    sum_of_weights = top_k_similar_users.sum()
    
    if sum_of_weights == 0:
        return 0.0
        
    predicted_rating = weighted_sum / sum_of_weights
    
    # Ограничиваем предсказание рамками [-1, 1]
    return np.clip(predicted_rating, -1, 1)

# --- 5. Гибридная рекомендательная функция ---

def get_hybrid_recommendations(user_id, recipes_df, interactions_df, 
                               cb_cosine_sim, item_id_to_index_map, 
                               cf_user_item_matrix, cf_user_sim_df,
                               n=5, alpha=0.5):
    """
    Генерирует гибридные рекомендации для пользователя.
    
    :param cf_user_item_matrix: Матрица "пользователь-рецепт" из CF
    :param cf_user_sim_df: Матрица схожести пользователей из CF
    ... (остальные параметры) ...
    """
    print(f"\n--- Генерация гибридных рекомендаций для user_id={user_id} ---")
    print(f"Веса: CF(kNN)={alpha*100}%, CB(Feature Fusion)={(1-alpha)*100}%")

    # 1. Получаем список "понравившихся" и "уже просмотренных" пользователем рецептов
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
            # Усреднение профиля пользователя по его понравившимся рецептам
            user_cb_profile = cb_cosine_sim[liked_indices].mean(axis=0)
        else:
            user_cb_profile = np.zeros(len(item_id_to_index_map))
    else:
        user_cb_profile = np.zeros(len(item_id_to_index_map))

    # 4. Расчет гибридной оценки для каждого не просмотренного рецепта
    recommendations = []
    
    # Нормализуем предсказания CF (kNN) к [0, 1]
    # (x - min) / (max - min) -> (x - (-1)) / (1 - (-1)) -> (x + 1) / 2
    # Шкала CB (cosine_sim) уже [0, 1]

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
        # Нормализация CF оценки
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

# --- 6. Основной блок выполнения ---

def start_fun(n, top=5, _RECIPES_FILE_PATH = "dataset/data.csv", _INTERACTIONS_FILE_PATH = "dataset/interaction.csv"):
    
    # --- УКАЖИТЕ ПУТИ К ВАШИМ ФАЙЛАМ ---
    # Замените "recipes.csv" и "interactions.csv" на реальные пути к вашим файлам
    RECIPES_FILE_PATH = _RECIPES_FILE_PATH
    INTERACTIONS_FILE_PATH = _INTERACTIONS_FILE_PATH
    # Например: RECIPES_FILE_PATH = "C:/Users/Admin/Documents/my_recipes.csv"
    # ---
    
    # 1. Загрузка и подготовка данных
    # print(f"Загрузка данных из {RECIPES_FILE_PATH} и {INTERACTIONS_FILE_PATH}...")
    recipes_df, interactions_df = load_and_preprocess_data(RECIPES_FILE_PATH, INTERACTIONS_FILE_PATH)
    
    # Проверка, что данные загрузились
    # if recipes_df.empty or interactions_df.empty:
    #     print("Данные не загружены. Проверьте пути к файлам и их содержимое. Выход.")
    #     exit()

    # 2. Создание карты {item_id: index}
    item_id_to_index_map = pd.Series(recipes_df.index, index=recipes_df['id']).to_dict()

    # 3. Обучение моделей
    # Модель CF (kNN)
    cf_user_item_matrix, cf_user_sim_df = build_cf_components(interactions_df)
    # Модель CB (Feature Fusion)
    cb_cosine_sim_matrix = build_cb_model(recipes_df)

    # --- Демонстрация ---
    
    TEST_USER_ID = n
    
    user_likes = interactions_df[
        (interactions_df['user_id'] == TEST_USER_ID) & 
        (interactions_df['interaction'] == 1)
    ]['item_id'].tolist()
    
    # Добавлен более надежный фильтр на случай, если user_id не найден
    # if TEST_USER_ID not in interactions_df['user_id'].unique():
    #     print(f"Тестовый пользователь с ID {TEST_USER_ID} не найден в данных.")
    #     exit()
        
    liked_recipes = recipes_df[recipes_df['id'].isin(user_likes)]['Name_recipe'].tolist()
    print(f"\nПользователю {TEST_USER_ID} понравились: {liked_recipes}")

    # Получаем рекомендации с рекомендованным коэффициентом alpha = 0.3
    # Основание: При небольшой и разреженной матрице взаимодействий (как в начале работы системы),
    # Content-Based модель (70%) более надежна, чем Collaborative Filtering (30%).
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
        alpha=RECOMMENDED_ALPHA  # 30% CF, 70% CB - Рекомендованный коэффициент
    )
    
    # print(f"\n--- Итоговые гибридные рекомендации (alpha={RECOMMENDED_ALPHA}) с улучшенным CB: ---")
    # print("Рекомендация: CB (70%) теперь использует Feature Fusion (Tags 40%, Ingredients 30%, Name/Desc 30%).")
    # print(hybrid_recs_03)
    return hybrid_recs_03