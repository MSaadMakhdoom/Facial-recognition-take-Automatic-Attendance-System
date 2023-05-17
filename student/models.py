from django.db import models
import os
import uuid
# Create your models here.
from django.contrib.auth.models import User

class Class(models.Model):
    name = models.CharField(max_length=100)


    def __str__(self):
        return self.name
    

class Teacher(models.Model):
    name = models.CharField(max_length=50)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    teacher_class = models.ForeignKey(Class, on_delete=models.CASCADE)

    def __str__(self):
        return self.name
    

class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_number = models.IntegerField()
    email = models.EmailField(default='std@gmail.com', unique=True, null=True, blank=True)
    password = models.CharField(max_length=50,default='1234',null=True, blank=True)
    class_name = models.ForeignKey(Class, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/', null=True, blank=True) 
    def __str__(self):
        return self.name

class Attendance(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendance_records')
    
    status = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.student.name} - {self.date}: {self.status}"
