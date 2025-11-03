from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Scalable_Deep_Learning_for_Categorization_of_Satellite_Images

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
 


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})



# def DatasetView(request):
#     path = settings.MEDIA_ROOT + "//" + 'data.csv'
#     df = pd.read_csv(path, nrows=100)
#     df = df.to_html
#     return render(request, 'users/viewdataset.html', {'data': df})


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

 

def ml(request):
    
    # Create an empty dataframe
    data=pd.DataFrame()
    data = pd.DataFrame(columns=['image_path', 'label'])

    labels = {
        os.path.join(settings.MEDIA_ROOT, 'data', 'cloudy'): 'Cloudy',
        os.path.join(settings.MEDIA_ROOT, 'data', 'desert'): 'Desert',
        os.path.join(settings.MEDIA_ROOT, 'data', 'green_area'): 'Green_Area',
        os.path.join(settings.MEDIA_ROOT, 'data', 'water'): 'Water',
    }

    for folder in labels:
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            label = labels[folder]
            new_data = pd.DataFrame({'image_path': image_path, 'label': label}, index=[0])
            data = pd.concat([data, new_data])

    # Save the data to a CSV file
    csv_path = settings.MEDIA_ROOT + '//'  + 'image_dataset1.csv'      
    data.to_csv(csv_path, index=False)
      
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

    df =pd.read_csv(settings.MEDIA_ROOT + '//'  + 'image_dataset1.csv')      

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Pre-process the data
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       rotation_range=45,
                                       vertical_flip=True,
                                       fill_mode='nearest')


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        x_col="image_path",
                                                        y_col="label",
                                                        target_size=(255, 255),
                                                        batch_size=42,
                                                        class_mode="categorical")

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      x_col="image_path",
                                                      y_col="label",
                                                      target_size=(255, 255),
                                                      batch_size=42,
                                                      class_mode="categorical") 
    
    # Build a deep learning model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), input_shape=(253, 253, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    path = settings.MEDIA_ROOT + '//' + 'Model.h5'
    model = load_model(path)
    num_samples = test_df.shape[0]
    score = model.evaluate(test_generator, steps=num_samples // 42 + 1)

    # Get predictions
    y_pred = model.predict(test_generator, steps=num_samples // 42 + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calculate and print classification report
    report = classification_report(true_classes, y_pred_classes, target_names=class_labels)
    print("Classification Report:\n", report)

    # Calculate and plot confusion matrix
    cm = confusion_matrix(true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png'))
    plt.close()

    # Render the results to an HTML template
    return render(
        request,
        'users/ml.html',
        {
            'accuracy': score[1],  # Assuming accuracy is at index 1 in score list
            'classification_report': report,
            'confusion_matrix_image': '/media/confusion_matrix.png',  # Replace with your path
        }
    )


def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        # Load the trained model
        model_path = settings.MEDIA_ROOT + '/Model.h5'
        model = load_model(model_path)
        
        class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage(location='media/images/')
        img_path = fs.save(uploaded_image.name, uploaded_image)
        print(img_path)
        print('File uploaded successfully.')

        folder_path = os.path.join(settings.MEDIA_ROOT, 'images')
        n = uploaded_image.name
        image = os.path.join(folder_path,n)
        img = load_img(image, target_size=(255, 255))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_label = class_names[class_index]
        print("The image is predicted to be '{}'.".format(predicted_label))
        render_image = folder_path + '\\' + n

        context = {
            'predicted_image':render_image,
            'predicted_label':predicted_label,
        }
        
        # Return the predicted label as response
        return render(request, 'users/prediction.html',context)

    return render(request, 'users/predictForm.html', {})

 
 