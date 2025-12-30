"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    # path('', views.home),
    path('settings/', views.settings_page, name='settings'),
    # path('admin/', admin.site.urls),
    
    # path('', views.csv_display_view),
    path('test', views.test),
    path('', views.recipe_list, name='recipe_list'), # recipe_list или home_index
    path('whoami', views.whoami, name='whoami'),
    path("recipe/<int:id>", views.card),
    path('admin/', admin.site.urls),
    path('reg/', views.reg, name='reg'),
    path('login/', views.login_page, name='login'),
    path('logout/', views.logout_page, name='logout'),
    path('api/', views.process_button, name='process_button'),
    # path('api/reg/', views.reg, name='api_reg'),
    # path('logout/', views.logout_view, name='logout'),
    # path('all_recipes/', views.all_recipes, name='all_recipes'),
    path('api/reaction/', views.handle_reaction, name='handle_reaction'),
    path('api/react/', views.react_to_recipe, name='react_to_recipe'),
    path('search', views.search_recipes, name='search_recipes'),
    path('fridge/', views.fridge_page),
    path('about/', views.about_page),
]
