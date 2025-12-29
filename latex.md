# Математическая модель системы рекомендаций блюд

## 1. Контентно-ориентированная фильтрация (Content-Based)

### 1.1. Векторизация текстовых признаков

Для каждого рецепта $i$ формируется объединённый текстовый дескриптор:

$$
D_i = w_{\text{name}} \cdot \text{Name}_i + w_{\text{desc}} \cdot \text{Desc}_i + w_{\text{tags}} \cdot \text{Tags}_i + w_{\text{ingr}} \cdot \text{Ingredients}_i
$$

где:
- $\text{Name}_i$, $\text{Desc}_i$ — название и описание рецепта
- $\text{Tags}_i$ — список тегов
- $\text{Ingredients}_i$ — список ингредиентов
- $w_{\text{name}} = 0.3$, $w_{\text{tags}} = 0.4$, $w_{\text{ingr}} = 0.3$ — весовые коэффициенты

### 1.2. TF-IDF векторизация

Каждый дескриптор $D_i$ преобразуется в вектор:

$$
\mathbf{v}_i = \text{TF-IDF}(D_i)
$$

где TF-IDF — функция векторизации с косинусной нормализацией.

### 1.3. Матрица контентного сходства

Попарное сходство рецептов вычисляется через косинусную меру:

$$
\text{sim}_{\text{content}}(i, j) = \cos(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \cdot \|\mathbf{v}_j\|}
$$

Итоговая взвешенная матрица:

$$
\mathbf{S}_{\text{content}} = 0.3 \cdot \mathbf{S}_{\text{name}} + 0.4 \cdot \mathbf{S}_{\text{tags}} + 0.3 \cdot \mathbf{S}_{\text{ingr}}
$$

где $\mathbf{S}_{\text{name}}$, $\mathbf{S}_{\text{tags}}$, $\mathbf{S}_{\text{ingr}}$ — матрицы сходства для отдельных признаков.

---

## 2. Модель пользовательских предпочтений

### 2.1. Вектор интересов пользователя

$$
\mathbf{U} = \frac{1}{|L|} \sum_{r \in L} \mathbf{S}_{\text{content}}(r, :) - \frac{0.5}{|D|} \sum_{r \in D} \mathbf{S}_{\text{content}}(r, :)
$$

где:
- $L$ — множество лайкнутых рецептов
- $D$ — множество дизлайкнутых рецептов
- $\mathbf{S}_{\text{content}}(r, :)$ — строка матрицы сходства для рецепта $r$
- Коэффициент $0.5$ ослабляет влияние дизлайков (negative sampling)

### 2.2. Базовый скор рецепта

Для каждого кандидатного рецепта $c$:

$$
\text{score}_{\text{base}}(c) = \mathbf{U}[c]
$$

---

## 3. Популярность (для холодного старта)

### 3.1. Логарифмированный скор популярности

$$
\text{popularity}_{\text{raw}}(r) = \log(1 + \text{Likes}(r) + \text{Saves}(r))
$$

где $\text{Likes}(r)$, $\text{Saves}(r)$ — количество лайков и сохранений рецепта $r$.

### 3.2. Нормализация

$$
\text{popularity}_{\text{norm}}(r) = \frac{\text{popularity}_{\text{raw}}(r) - \min}{\max - \min}
$$

где $\min$, $\max$ — минимальное и максимальное значения в датасете.

---

## 4. Фактор "Холодильника"

### 4.1. Коэффициент совпадения

Для рецепта $r$ с множеством ингредиентов $I_r$ и холодильника пользователя $F$:

$$
\text{overlap}(r, F) = \frac{|I_r \cap F|}{|I_r|}
$$

где $|\cdot|$ — мощность множества.

### 4.2. Мультипликативный буст

$$
\text{boost}(r) = 1 + \text{overlap}(r, F) \in [1, 2]
$$

Рецепт, все ингредиенты которого есть у пользователя, получает удвоенный скор.

---

## 5. Гибридная модель

### 5.1. Итоговый скор

Для пользователей с историей:

$$
\text{score}_{\text{final}}(r) = \text{score}_{\text{base}}(r) \cdot \text{boost}(r)
$$

Для холодного старта:

$$
\text{score}_{\text{final}}(r) = \text{popularity}_{\text{norm}}(r) \cdot \text{boost}(r)
$$

### 5.2. Условия применения

$$
\text{score}_{\text{final}}(r) = 
\begin{cases}
\text{popularity}_{\text{norm}}(r) \cdot \text{boost}(r), & \text{если } L = \emptyset \text{ и } F \neq \emptyset \\
\text{popularity}_{\text{norm}}(r), & \text{если } L = \emptyset \text{ и } F = \emptyset \\
\text{score}_{\text{base}}(r) \cdot \text{boost}(r), & \text{иначе}
\end{cases}
$$

---

## 6. Выдача рекомендаций

### 6.1. Отбор кандидатов

$$
\text{Candidates} = \text{Top}_{3N}(\text{score}_{\text{final}}) \setminus \text{Seen}
$$

где:
- $N$ — требуемое количество рекомендаций
- $\text{Seen}$ — множество уже просмотренных рецептов
- $\text{Top}_{3N}$ — топ $3N$ рецептов по убыванию $\text{score}_{\text{final}}$

### 6.2. Стратегия Random Offset

$$
\begin{aligned}
N_{\text{fixed}} &= N \cdot (1 - \alpha) \\
N_{\text{random}} &= N \cdot \alpha
\end{aligned}
$$

где $\alpha \in [0, 1]$ — параметр случайности (по умолчанию $\alpha = 0.2$).

### 6.3. Алгоритм выдачи

1. Выбрать $N_{\text{fixed}}$ топовых рецептов из $\text{Candidates}$
2. Случайно выбрать $N_{\text{random}}$ из оставшихся кандидатов
3. Перемешать финальный список:

$$
\text{Recommendations} = \text{Shuffle}(\text{Top}_{N_{\text{fixed}}} \cup \text{Random}_{N_{\text{random}}})
$$

---

## 7. Итоговая формула системы

### Для пользователей с историей:

$$
\text{Recommendation}(r) = \left[ \frac{1}{|L|} \sum_{l \in L} \mathbf{S}_{\text{content}}(l, r) - \frac{0.5}{|D|} \sum_{d \in D} \mathbf{S}_{\text{content}}(d, r) \right] \cdot \left( 1 + \frac{|I_r \cap F|}{|I_r|} \right)
$$

### Для новых пользователей:

$$
\text{Recommendation}(r) = \text{popularity}_{\text{norm}}(r) \cdot \left( 1 + \frac{|I_r \cap F|}{|I_r|} \right)
$$

---

## Ключевые особенности модели

1. **Гибридность**: Content-Based + Popularity + Collaborative (через дизлайки)
2. **Обработка cold start**: Через популярность и ингредиенты холодильника
3. **Negative sampling**: Учет дизлайков с коэффициентом 0.5
4. **Практичность**: Мультипликативный буст для ингредиентов
5. **Диверсификация**: Механизм random offset предотвращает застой
6. **Взвешивание признаков**: Название (30%), теги (40%), ингредиенты (30%)


$$Multiplier = 1 + \frac{Matches}{TotalIngredients}$$
