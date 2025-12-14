# from django.contrib.auth.backends import ModelBackend
# from django.contrib.auth import get_user_model
# from django.db.models import Q

# User = get_user_model()

# class EmailOrUsernameBackend(ModelBackend):
#     """
#     Аутентификация по email ИЛИ username
#     """
    
#     def authenticate(self, request, username=None, password=None, **kwargs):
#         try:
#             # Ищем пользователя по email или username
#             user = User.objects.get(
#                 Q(email=username) | Q(username=username)
#             )
#         except User.DoesNotExist:
#             return None
        
#         # Проверяем пароль
#         if user.check_password(password):
#             return user
#         return None
    
#     def get_user(self, user_id):
#         try:
#             return User.objects.get(pk=user_id)
#         except User.DoesNotExist:
#             return None