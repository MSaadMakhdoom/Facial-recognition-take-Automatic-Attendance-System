from django.urls import path
from . import views
app_name = 'student'
urlpatterns = [
    path('',views.Home_Page, name='home_page'),

    path('login/', views.student_login, name='student_login'),
    path('logout/', views.logout_view, name='logout'),
    path('teacher-login/',views.teacher_login_view,name ='teacher_login'),
    path('teacher-view-attendence-report/',views.teacher_view_attendance,name ='teacher_view_attendence_report'),
    path('automate-attendence/',views.Home_Page_Automated_Attendence, name='automate-attendence'),
    
    path('take-attendence/',views.take_attendence, name='take-attendence'),
    path('attendence/',views.run, name='attendance')
]
