from django.contrib import admin

# Register your models here.
from .models import Class, Student, Attendance,Teacher

admin.site.register(Class)
admin.site.register(Student)
admin.site.register(Attendance)
admin.site.register(Teacher)