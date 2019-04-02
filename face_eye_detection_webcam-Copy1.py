#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[ ]:





# In[2]:


import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


img1 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/1.jpg',0)


# In[4]:


img2 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/2.jpg',0)


# In[5]:


img3 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/3.jpg',0)


# In[6]:


img4 = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/4.jpg',0)


# In[7]:


plt.imshow(img2,cmap='gray')


# In[8]:


plt.imshow(img3)


# In[ ]:





# In[9]:


face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml')


# In[10]:


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),5)
        return face_img


# In[11]:


result1 = detect_face(img2)


# In[12]:


plt.imshow(result1)


# In[13]:


eye_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_eye.xml')


# In[14]:


def detect_eyes(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),2)
        return face_img


# In[15]:


result2 = detect_eyes(img2)


# In[16]:


plt.imshow(result2,cmap='gray')


# In[17]:


#nowforlivecam


# In[18]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_default.xml')


# In[20]:


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        return face_img


# In[21]:


eye_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_eye.xml')


# In[22]:


def detect_eyes(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        return face_img


# In[ ]:


cap = cv2.VideoCapture(0)
while True:
    
    ret,frame = cap.read(0)
    
    frame = detect_face(frame)
    frame1 = detect_eyes(frame)
    cv2.imshow('Video Face Detect',frame)
    cv2.imshow('Video Face Detect',frame1)
    
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
   


# In[ ]:




