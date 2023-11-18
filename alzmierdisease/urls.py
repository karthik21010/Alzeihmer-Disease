"""alzmierdisease URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from alzmierdisease import views as alzmr
from patient import views as patient
from doctor import views as doctor

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('',alzmr.index, name="index"),
    path('base/', alzmr.base, name="base"),
    path('admin1/', alzmr.adminlogin, name="admin1"),
    path('adminloginaction/', alzmr.adminloginaction, name="adminloginaction"),
    path('activatepatient/',alzmr.activatepatient,name='activatepatient'),
    path('testing/', alzmr.testing, name="testing"),
    path('svm/', alzmr.svm, name="svm"),
    path('svm1/', alzmr.svm1, name="svm1"),
    path('logout/', alzmr.logout, name="logout"),
    path('svm11/',alzmr.svm11,name='svm11'),


    path('doctorlogin/',doctor.doctorlogin,name='doctorlogin'),
    path('doctorregister/',doctor.doctorregister,name='doctorregister'),
    path('doctordetails/', doctor.doctordetails, name='doctordetails'),
    path('activatedoctor/',doctor.activatedoctor,name='activatedoctor'),
    path('doctorlogincheck/',doctor.doctorlogincheck,name='doctorlogincheck'),
    path('patientdata/',doctor.doctorviewpatientdata,name='patientdata'),
    path('addtreatment/',doctor.addtreatment,name='addtreatment'),
    path('report/',doctor.report,name='report'),



    path('patientlogin/', patient.patientlogin, name='patientlogin'),
    path('patientregister/', patient.patientregister, name='patientregister'),
    path('patientlogincheck/',patient.patientlogincheck,name='patientlogincheck'),
    path('patientdetails/', patient.patientdetails, name='patientdetails'),
    path('symptoms/', patient.patientsymptoms, name='symptoms'),
    path('patntsymptms/', patient.patntsymptms, name='patntsymptms'),


]


