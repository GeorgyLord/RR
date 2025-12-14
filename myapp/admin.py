from django.contrib import admin

# Register your models here.
from .models import Recipe
from .models import CustomUser
# from .models import Interactions

# @admin.register(Interactions)
# class InteractionsAdmin(admin.ModelAdmin):
#     # class Meta:
#     #     db_table = 'tag_table'
#     list_display = [field.name for field in Interactions._meta.fields]
#     list_filter = []
#     search_fields = [] # Поиск по текстовым полям
#     ordering = ('-id',) # Сортировка по существующему полю

@admin.register(CustomUser)
class UserAdmin(admin.ModelAdmin):
    # class Meta:
    #     db_table = 'tag_table'
    list_display = [field.name for field in CustomUser._meta.fields]
    list_filter = []
    search_fields = [] # Поиск по текстовым полям
    ordering = ('-id',) # Сортировка по существующему полю


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    # class Meta:
    #     verbose_name = 'Пользователь'
    #     verbose_name_plural = 'Пользователи'
    list_display = [field.name for field in Recipe._meta.fields]
    list_filter = []
    search_fields = ['name', 'description'] # Поиск по текстовым полям
    ordering = ('-id',) # Сортировка по существующему полю