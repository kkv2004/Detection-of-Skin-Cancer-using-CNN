import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import io
import seaborn as sns
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras.models import load_model

# Function to create and train the skin cancer detection model
def train_skin_cancer_model(train_dir, validation_dir,test_dir, epochs=5):
                batch_size=32
                train_datagen = ImageDataGenerator(
                    rescale=1.0 / 255.0,
                    rotation_range=40,  
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

                validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()

                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(224, 224),  
                    batch_size=batch_size,
                    class_mode='binary'
                    
                )

                validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(224, 224),  
                    batch_size=batch_size,
                    class_mode='binary'
                    
                )
                sys.stdout = original_stdout


                model = Sequential([
                    Conv2D(32, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    Conv2D(64, (5, 5), activation='relu'),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    Conv2D(64, (3, 3), activation='relu'),
                    Conv2D(64, (3, 3), activation='relu'),
                    Conv2D(128, (3, 3), activation='relu'),
                    MaxPooling2D((3, 3), strides=(2, 2)),
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(4096, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])



                # Compile the model
                model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])  # Adjust learning rate

                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='accuracy',  
                    patience=5,
                    restore_best_weights=True)

                # Train the model
                history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator,callbacks=[early_stopping], verbose=2)
                # Function to evaluate the model on test data
                
                test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

                # Create a generator for test data
                test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size=(224, 224),
                                    batch_size=32,  
                                    class_mode='binary',
                                    shuffle=False) 

                model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
                
                test_accuracy = model.evaluate(test_generator,verbose=0)
                y_true = test_generator.classes
                y_pred = model.predict(test_generator)
                model.save("skin_detection_model.keras")
                return model,test_accuracy,y_pred,y_true

def preprocess_image(image_path):
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))
                image = image.astype('float32') / 255.0
                return image

def detect_skin_cancer(image_path):
                model=load_model("skin_detection_model.keras")
                preprocessed_image = preprocess_image(image_path)
                prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
                cancer_probability = prediction[0][0]  
                threshold = 0.5  
                if cancer_probability > threshold:
                                return "The image likely contains cancer."
                else:
                                return "The image is likely normal."
                
                



# Define the folder path containing the images
train_dir=r"C:\Users\sekar\Desktop\CNNPROJECT\New Data\train"
test_dir=r"C:\Users\sekar\Desktop\CNNPROJECT\New Data\test"
validation_dir=r"C:\Users\sekar\Desktop\CNNPROJECT\New Data\validation"


flag=True
# Train the model
while flag==True:
                print("MENU")
                print("1.Training and Testing")
                print("2.Cancer Detection")
                print("3.Exit")
                ch=input("Enter the choice:")
                if(ch=='1'):
                                model,test_accuracy,y_pred,y_true=train_skin_cancer_model(train_dir, validation_dir,test_dir)
                                
                                # Evaluate the model on test data
                                print("Test Accuracy:", test_accuracy * 100)
                                y_pred_binary = (y_pred > 0.5).astype(int)

                                # Calculate confusion matrix
                                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

                                # Calculate sensitivity
                                sensitivity = tp/(tp + fn)
                                print("Sensitivity (True Positive Rate):", sensitivity)
                                precision = tp/(tp+fp)

                                # Print the calculated precision
                                print("Precision:", precision)
                                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

                                # Print the calculated F1 score
                                print("F1 Score:", f1_score)

                                metrics = ['Accuracy', 'Sensitivity', 'Precision', 'F1 Score']
                                '''
                                #values = [test_accuracy, sensitivity, precision, f1_score]

                                plt.figure(figsize=(8, 6))
                                plt.bar(metrics, values, color='skyblue')
                                plt.ylim(0, 1)  # Set y-axis limit to 0-1 for probability values
                                plt.ylabel('Score')
                                plt.title('Model Evaluation Metrics')
                                plt.show()
                                '''
                                plt.figure(figsize=(8, 6))
                                plt.hist(y_pred, bins=20, color='skyblue', alpha=0.7)
                                plt.xlabel('Predicted Probabilities')
                                plt.ylabel('Frequency')
                                plt.title('Histogram of Predicted Probabilities')
                                plt.show()
                elif ch=='2':
                                
                                image_path = input("Enter the path to the image file: ")
                                result = detect_skin_cancer(image_path)
                                print(result)

                elif ch=='3':
                                sys.exit(0)
                else:
                                print("Enter the correct choice number")
