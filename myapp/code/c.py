# import pandas as pd
# import django
# import sys, os

# # Добавляем родительские директории в путь
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # до проекта
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # до приложения

# # Настраиваем Django
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
# django.setup()

# from myapp.models import Recipe  # замените your_app на имя вашего приложения

# def import_from_csv(csv_file_path):
#     # Читаем CSV файл
#     df = pd.read_csv(csv_file_path)
    
#     for index, row in df.iterrows():
#         try:
#             Recipe.objects.create(
#                 Id_Recipe=int(row['id']) if pd.notnull(row['id']) else None,
#                 URL=row['URL'] if pd.notnull(row['URL']) else '',
#                 Name_recipe=row['Name_recipe'] if pd.notnull(row['Name_recipe']) else '',
#                 Description=row['Description'] if pd.notnull(row['Description']) else '',
#                 Author=row['Author'] if pd.notnull(row['Author']) else '',
#                 Cooking_time=int(row['Cooking_time']) if pd.notnull(row['Cooking_time']) else 0,
#                 Likes=int(row['Likes']) if pd.notnull(row['Likes']) else 0,
#                 Dislikes=int(row['Dislikes']) if pd.notnull(row['Dislikes']) else 0,
#                 Safes=int(row['Safes']) if pd.notnull(row['Safes']) else 0,
#                 Type_recipe=row['Type_recipe'] if pd.notnull(row['Type_recipe']) else '',
#                 Tags=str(row['Tags']) if pd.notnull(row['Tags']) else '[]',
#                 Count_ingredients=int(row['Count_ingredients']) if pd.notnull(row['Count_ingredients']) else 0,
#                 Ingredients=str(row['Ingredients']) if pd.notnull(row['Ingredients']) else '[]',
#                 Pontions=int(row['Pontions']) if pd.notnull(row['Pontions']) else 0,
#                 Calorie_content=float(row['Calorie_content']) if pd.notnull(row['Calorie_content']) else 0,
#                 Squirrels=float(row['Squirrels']) if pd.notnull(row['Squirrels']) else 0,
#                 Fats=float(row['Fats']) if pd.notnull(row['Fats']) else 0,
#                 Carbohydrates=float(row['Carbohydrates']) if pd.notnull(row['Carbohydrates']) else 0,
#                 Steps_text=str(row['Steps_text']) if pd.notnull(row['Steps_text']) else '[]',
#                 Steps_images=str(row['Steps_images']) if pd.notnull(row['Steps_images']) else '[]',
#                 Url_steps_images=str(row['Url_steps_images']) if pd.notnull(row['Url_steps_images']) else '[]',
#                 Images_recipe=str(row['Images_recipe']) if pd.notnull(row['Images_recipe']) else '[]',
#                 Url_images_recipe=str(row['Url_images_recipe']) if pd.notnull(row['Url_images_recipe']) else '[]',
#                 Number_page=int(row['Number_page']) if pd.notnull(row['Number_page']) else 0,
#             )
#             print(f'Imported: {row["Name_recipe"]}')
#         except Exception as e:
#             print(f'Error importing row {index}: {e}')
#             continue

# # Использование
# if __name__ == '__main__':
#     # print(str(__path__))
#     import_from_csv('dataset/data.csv')







# myapp/code/c.py
import pandas as pd
import django
import sys
import os
import argparse

# Настройка путей
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from myapp.models import Recipe

def main():
    parser = argparse.ArgumentParser(description='Импорт рецептов из CSV в Django')
    parser.add_argument('--csv', type=str, default='data.csv', 
                       help='Путь к CSV файлу (по умолчанию: data.csv)')
    parser.add_argument('--clear', action='store_true',
                       help='Очистить базу перед импортом')
    
    args = parser.parse_args()
    
    # Определяем полный путь
    if os.path.isabs(args.csv):
        csv_path = args.csv
    else:
        # Пробуем разные варианты
        possible_paths = [
            os.path.join(project_root, args.csv),
            os.path.join(project_root, 'dataset', args.csv),
            args.csv
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        else:
            print(f"Файл {args.csv} не найден!")
            sys.exit(1)
    
    print(f"Импорт из: {csv_path}")
    
    # Очистка базы если нужно
    if args.clear:
        print("Очищаем базу данных...")
        Recipe.objects.all().delete()
    
    # Импорт
    try:
        df = pd.read_csv(csv_path)
        print(f"Найдено {len(df)} записей в CSV")
        
        recipes_to_create = []
        for index, row in df.iterrows():
            try:
                recipe = Recipe(
                    Id_Recipe=int(row['id']) if pd.notnull(row['id']) else None,
                    URL=row['URL'],
                    Name_recipe=row['Name_recipe'],
                    Description=row['Description'],
                    Author=row['Author'],
                    Cooking_time=int(row['Cooking_time']) if pd.notnull(row['Cooking_time']) else 0,
                    Likes=int(row['Likes']) if pd.notnull(row['Likes']) else 0,
                    Dislikes=int(row['Dislikes']) if pd.notnull(row['Dislikes']) else 0,
                    Safes=int(row['Safes']) if pd.notnull(row['Safes']) else 0,
                    Type_recipe=row['Type_recipe'],
                    Tags=str(row['Tags']) if pd.notnull(row['Tags']) else '[]',
                    Count_ingredients=int(row['Count_ingredients']) if pd.notnull(row['Count_ingredients']) else 0,
                    Ingredients=str(row['Ingredients']) if pd.notnull(row['Ingredients']) else '[]',
                    Pontions=int(row['Pontions']) if pd.notnull(row['Pontions']) else 0,
                    Calorie_content=float(row['Calorie_content']) if pd.notnull(row['Calorie_content']) else 0,
                    Squirrels=float(row['Squirrels']) if pd.notnull(row['Squirrels']) else 0,
                    Fats=float(row['Fats']) if pd.notnull(row['Fats']) else 0,
                    Carbohydrates=float(row['Carbohydrates']) if pd.notnull(row['Carbohydrates']) else 0,
                    Steps_text=str(row['Steps_text']) if pd.notnull(row['Steps_text']) else '[]',
                    Steps_images=str(row['Steps_images']) if pd.notnull(row['Steps_images']) else '[]',
                    Url_steps_images=str(row['Url_steps_images']) if pd.notnull(row['Url_steps_images']) else '[]',
                    Images_recipe=str(row['Images_recipe']) if pd.notnull(row['Images_recipe']) else '[]',
                    Url_images_recipe=str(row['Url_images_recipe']) if pd.notnull(row['Url_images_recipe']) else '[]',
                    Number_page=int(row['Number_page']) if pd.notnull(row['Number_page']) else 0,
                )
                recipes_to_create.append(recipe)
                
                if index % 100 == 0:
                    print(f"Подготовлено {index + 1} записей...")
                    
            except Exception as e:
                print(f"Ошибка в строке {index}: {e}")
                continue
        
        # Массовое создание
        if recipes_to_create:
            Recipe.objects.bulk_create(recipes_to_create)
            print(f"Успешно создано {len(recipes_to_create)} записей!")
            
    except Exception as e:
        print(f"Ошибка при импорте: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()