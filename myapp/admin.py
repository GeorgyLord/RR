from django.contrib import admin

# Register your models here.
from .models import Recipe
from .models import Person

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = [field.name for field in Person._meta.fields]
    list_filter = []
    search_fields = [] # Поиск по текстовым полям
    ordering = ('-id',) # Сортировка по существующему полю


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = [field.name for field in Recipe._meta.fields]
    list_filter = []
    search_fields = ['name', 'description'] # Поиск по текстовым полям
    ordering = ('-id',) # Сортировка по существующему полю