from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Recipe

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Логин'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control',
        'placeholder': 'Пароль'
    }))
    
    
    
    
    
class RecipeForm(forms.ModelForm):
    # Скрытые поля для шагов и изображений
    steps = forms.CharField(widget=forms.HiddenInput(), required=False)
    step_images = forms.CharField(widget=forms.HiddenInput(), required=False)
    
    # Убираем Author из формы - будем заполнять автоматически
    # Number_page делаем необязательным с значением 0 по умолчанию
    
    # Настройка полей по умолчанию
    Cooking_time = forms.IntegerField(
        initial=1,
        min_value=1,
        widget=forms.NumberInput(attrs={'min': 1})
    )
    
    Portions = forms.IntegerField(
        initial=1,
        min_value=1,
        widget=forms.NumberInput(attrs={'min': 1})
    )
    
    # Number_page - делаем необязательным с default=0
    Number_page = forms.IntegerField(
        initial=0,
        min_value=0,
        required=False,  # Важно: делаем необязательным
        widget=forms.NumberInput(attrs={'min': 0})
    )
    
    Calorie_content = forms.FloatField(
        initial=0.0,
        min_value=0.0,
        widget=forms.NumberInput(attrs={'min': 0, 'step': 0.1})
    )
    
    Squirrels = forms.FloatField(
        initial=0.0,
        min_value=0.0,
        widget=forms.NumberInput(attrs={'min': 0, 'step': 0.1})
    )
    
    Fats = forms.FloatField(
        initial=0.0,
        min_value=0.0,
        widget=forms.NumberInput(attrs={'min': 0, 'step': 0.1})
    )
    
    Carbohydrates = forms.FloatField(
        initial=0.0,
        min_value=0.0,
        widget=forms.NumberInput(attrs={'min': 0, 'step': 0.1})
    )
    
    Ingredients = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Например:\n200:г:Мука\n2:шт:Яйца\n1:стакан:Молоко'
        })
    )
    
    class Meta:
        model = Recipe
        # Убираем Author из списка полей формы - заполним в view
        fields = [
            'Name_recipe',
            'Description',
            # 'Author',  # Убираем - будет заполняться автоматически
            'Cooking_time',
            'Type_recipe',
            'Tags',
            'Ingredients',
            'Portions',
            'Calorie_content',
            'Squirrels',
            'Fats',
            'Carbohydrates',
            'Number_page'  # Оставляем, но делаем необязательным
        ]
        
        labels = {
            'Name_recipe': 'Название рецепта',
            'Description': 'Описание',
            'Cooking_time': 'Время приготовления (минуты)',
            'Type_recipe': 'Тип рецепта',
            'Tags': 'Теги (через запятую)',
            'Ingredients': 'Ингредиенты',
            'Portions': 'Количество порций',
            'Calorie_content': 'Калории (ккал)',
            'Squirrels': 'Белки (г)',
            'Fats': 'Жиры (г)',
            'Carbohydrates': 'Углеводы (г)',
            'Number_page': 'Номер страницы (опционально)'
        }
        
        widgets = {
            'Description': forms.Textarea(attrs={'rows': 3}),
            'Tags': forms.TextInput(attrs={'placeholder': 'Завтрак, Обед, Десерт...'}),
            'Type_recipe': forms.Select(choices=[
                ('', 'Выберите тип'),
                ('Завтрак', 'Завтрак'),
                ('Обед', 'Обед'),
                ('Ужин', 'Ужин'),
                ('Десерт', 'Десерт'),
                ('Закуска', 'Закуска'),
                ('Напиток', 'Напиток'),
                ('Основное блюдо', 'Основное блюдо'),
                ('Суп', 'Суп'),
                ('Салат', 'Салат'),
            ]),
        }