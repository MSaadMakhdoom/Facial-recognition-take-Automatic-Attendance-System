from django.shortcuts import render, redirect
from .models import Class, Student, Attendance,Teacher
# Create your views here.
from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect
from django.urls import reverse
import cv2
from django.conf import settings
import os
import face_recognition
from datetime import datetime,timedelta,date
import numpy as np
from django.utils import timezone
from skimage import feature as detector
from django.core.mail import send_mail
from django.http import HttpResponse



# -----------------------------------------------------------------------------------------


def load_all_students():
    std_img = []
    std_id = []
    print("Start Load Student Images")
    students = Student.objects.all()
    print("Load Student Images")
    for student in students:
        print(student.id)
        img_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        print(img_path)
        
        # Check if the image path is valid and skip if not
        if os.path.isfile(img_path):
            # Load the image file using face_recognition library
            image = face_recognition.load_image_file(img_path)
            print(image)
            std_id.append(student.id)
            std_img.append(image)
        else:
            print("Invalid image path:", img_path)
        
    print("Finish Load Student Images")
    return std_id, std_img


def preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply histogram equalization
    eq = cv2.equalizeHist(blur)

    return eq

def preprocess_images(images, target_size):
    preprocessed_images = []
    for image in images:
        # Resize the image
        resized_image = cv2.resize(image, target_size)
        # Append the preprocessed image to the list of preprocessed images
        preprocessed_image = preprocess(resized_image)
        # Append the preprocessed image to the list of preprocessed images
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images

def extract_image_features(preprocessed_face_images):
    hog = cv2.HOGDescriptor()
    hog_features = []

    for im in preprocessed_face_images:
        if im.shape[0] < 8 or im.shape[1] < 8:
            continue  # Skip images that are smaller than the cell size

        try:
            features = detector.hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8))
            hog_features.append(features)
        except ValueError:
            continue  # Skip images that cause a ValueError

    hog_features = np.array(hog_features)
    print(len(hog_features))
    
    return hog_features



# -----------------------------------------------------------------------------------------

def FaceEncoding_img( imges):
    print("Start Encoding Image ")
    FaceencodeList = []
    for img in imges:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_recognition.face_encodings(img)[0]
        FaceencodeList.append(face)
    print("Finish Econding images")
    return FaceencodeList


def run(request):
    std_id,std_img = load_all_students()
    encodeListknown = FaceEncoding_img(std_img)
    cap = cv2.VideoCapture(0)
    print("start video capture")
    while True:
        success, img = cap.read()
        print("resize image")
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)
        print("Start Matching the face ")
        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            print("compare")
            face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            print("find distance")
            matchindex = np.argmin(face_distence)

            if matches_face[matchindex]:
                print("Face match database image")
                name = std_id[matchindex]
                y1, x2, y2, x1 = faceloc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print("mark attendence ")
                add_attendance(request,name,True)

        # cv2.imshow("campare", img)

        key = cv2.waitKey(1)
        if key == 27:
           break
    # Release the camera.
    cap.release()
    cv2.destroyAllWindows()

    # Return the HTML page.
    return render(request, 'take_attendence.html')


std_id, std_img = load_all_students()
encodeListknown = FaceEncoding_img(std_img)




def take_attendence(request):
    if request.method == 'POST':

        # Get the video stream from the frontend
        video_stream = request.FILES.get('video')

        print ('camera video stream',video_stream)

        video_path = os.path.join('/tmp', video_stream.name)
        with open(video_path, 'wb') as file:
            for chunk in video_stream.chunks():
                file.write(chunk)

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)
        print ('capture video ')
        # Continuously read frames from the video stream
        while True:
            success, img = cap.read()
            if not success:
                break

            # Resize and convert the frame to RGB
            imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)

            # Detect and encode the faces in the frame
            faces_current = face_recognition.face_locations(imgc)
            encode_faces_current = face_recognition.face_encodings(imgc, faces_current)

            # Compare the face encodings to the known faces
            for encodeFace, faceLoc in zip(encode_faces_current, faces_current):
                matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
                face_distances = face_recognition.face_distance(encodeListknown, encodeFace)
                match_index = np.argmin(face_distances)

                # If a match is found, mark the student's attendance as present
                if matches_face[match_index]:
                    name = std_id[match_index]
                    y1, x2, y2, x1 = faceLoc
                    # Mark attendance
                    add_attendance(request, name, True)

          

        # Release the VideoCapture object
        cap.release()

        return HttpResponse('Video received and processed successfully')

    return render(request, 'take_attendence.html')


#-----------------------------------------------------------------------------------------------


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_bounding_boxes(frame, faces):
    for (x, y, w, h) in faces:
        return cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def match_faces(frame):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('recognizer/trainingData.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence >= 45 and confidence <= 85:
            student = Student.objects.get(id=id_)
            cv2.putText(frame, student.name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def attendance(request):
    print("Automated Attendence System")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render(request, 'take_attendence.html', {'error': 'Could not open the camera.'})
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        draw_bounding_boxes(frame, faces)

        # Display the frame on the web page.
        # cv2.imshow('Attendance', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
           break
    # Release the camera.
    cap.release()
    cv2.destroyAllWindows()

    # Return the HTML page.
    return render(request, 'take_attendence.html')



def Home_Page(request):
    return render(request, 'main_page.html')

def Home_Page_Automated_Attendence(request):
    return render(request, 'base.html')
# ----------------------------------------------------------------------------------------------------------
# def take_attendence(request):
#     return render(request, 'take_attendence.html')



# ----------------------------------------------------------------------------------------------------------
def teacher_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse('student:automate-attendence'))
        else:
            return render(request, 'teacher_login.html', {'error': 'Invalid login credentials'})
    else:
        return render(request, 'teacher_login.html')
    

def add_attendance(request, student_id, status):
    print("For Attendence Student id", student_id)
    student = Student.objects.get(id=student_id)
    attendance_date = date.today()

    # Get the current time.
    now = datetime.now()

    # Check if the last attendance was taken more than 10 minutes ago.
    last_attendance = Attendance.objects.filter(student=student).order_by('-date').first()

    if last_attendance is None:
        print("No previous attendance found for this student.")
        # Then create a new Attendance instance with the student and current date.
        Attendance.objects.create(student=student,status =True)
        
        # Now, we send the email.
        # send_email_to_user(student,now)


    else:
        now = timezone.now()
        if (now - last_attendance.date) > timedelta(minutes=10):
            # Create a new Attendance instance with the student and current date.
            Attendance.objects.create(student=student,status =True)

            # Now, we send the email.
            #send_email_to_user(student,now)

        else:
            print("Attendance already taken for this student within the last 10 minutes.")


def send_email_to_user(user,now):
    print("Sending email to user")
    subject = "Attendance Marked Successfully"
    message = f"Dear {user.name},\n\nYour attendance has been marked successfully for today {now}.\n\nRegards,\nFAST"
    email_from = settings.EMAIL_HOST_USER
    print("email",user.email)
    recipient_list = [user.email,]

    send_mail(subject, message, email_from, recipient_list)

    print("Email sent successfully.")


def teacher_view_attendance(request):
    # Get the teacher's class
    teacher = Teacher.objects.get(user=request.user)
    print("teacher name",teacher)
    class_obj = teacher.teacher_class
    print("teacher class",class_obj)
    # Get all students in the teacher's class
    students = Student.objects.filter(class_name=class_obj)
    print("teacher student",students)
    # Get attendance records for all students in the class
    attendance_records = Attendance.objects.filter(student__class_name=class_obj)
    
    # Create a dictionary to store attendance data
    attendance_data = {}
    for student in students:
        attendance_data[student.name] = []
    
    # Populate attendance data dictionary with attendance records
    for record in attendance_records:
        attendance_data[record.student.name].append((record.date, record.status))
    
    print("class attendence record",attendance_data)
    # Render the attendance report template with the attendance data
    return render(request, 'teacher_attendance_report.html', {'attendance_data': attendance_data})


def student_login(request):
    if request.method == 'POST':
        # Get login form data and authenticate user
        email = request.POST['username']
        password = request.POST['password']
        print("Student Login",email,password)
        try:
            # Check if user exists in database
            student = Student.objects.get(email=email)
            print("student email",student)
        except Student.DoesNotExist:
            student = None
        # If user exists, check if password is correct
        try:
            student_password = Student.objects.get(password=password,email=email)
        except Student.DoesNotExist:
            student_password = None
        
        if student is not None and student_password is not None:
            # Credentials are correct, log in the user
            # filter attendace report student by email 
            attendance = Attendance.objects.filter(student=student)
            print("Attendance", attendance)

            return render(request, 'attendance_report.html', {'attendance': attendance})
        else:
            # Credentials are incorrect, show an error message
            error_message = "Invalid email or password. Please try again."
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')



def logout_view(request):
    logout(request)
    return redirect('student:teacher_login')