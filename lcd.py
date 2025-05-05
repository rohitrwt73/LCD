import numpy as np
import pandas as pd
import pydicom as dicom
import os
import cv2
from tkinter import *
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import sys
import tkinter.filedialog as filedialog

class TrainingProgressCallback(Callback):
    def __init__(self, progress_var, progress_label):
        super().__init__()
        self.progress_var = progress_var
        self.progress_label = progress_label
        
    def on_epoch_end(self, epoch, logs=None):
        progress = ((epoch + 1) / self.params['epochs']) * 100
        self.progress_var.set(progress)
        self.progress_label.config(
            text=f"Epoch {epoch + 1}/{self.params['epochs']} - "
                 f"Loss: {logs['loss']:.4f} - "
                 f"Accuracy: {logs['accuracy']:.4f}"
        )
        self.progress_label.update()

class LCD_CNN:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1006x600+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        img4 = Image.open(r"Images\Lung-Cancer-Detection.jpg").resize((1006, 500), Image.Resampling.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=50, width=1006, height=500)

        title_lbl = Label(text="Lung Cancer Detection", font=("Bradley Hand ITC", 30, "bold"), bg="black", fg="white")
        title_lbl.place(x=0, y=0, width=1006, height=50)

        self.progress_var = DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, 
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.place(x=80, y=550, width=846, height=20)
        
        self.progress_label = Label(
            self.root,
            text="Training Progress",
            font=("Times New Roman", 12)
        )
        self.progress_label.place(x=80, y=520, width=846, height=20)

        self.b1 = Button(text="Import Data", cursor="hand2", command=self.import_data, 
                        font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b1.place(x=80, y=130, width=180, height=30)
        
        self.b2 = Button(text="Pre-Process Data", cursor="hand2", command=self.preprocess_data, 
                        font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b2.place(x=80, y=180, width=180, height=30)
        self.b2["state"] = "disabled"
        
        self.b3 = Button(text="Train Data", cursor="hand2", command=self.train_data, 
                        font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b3.place(x=80, y=230, width=180, height=30)
        self.b3["state"] = "disabled"

        self.check_single_image()

    def import_data(self):
        try:
            self.dataDirectory = 'sample_images/'
            self.lungPatients = os.listdir(self.dataDirectory)
            self.labels = pd.read_csv('stage1_labels.csv', index_col=0)
            self.size = 32
            self.NoSlices = 5
            messagebox.showinfo("Import Data", "Data Imported Successfully!")
            self.b1["state"] = "disabled"
            self.b2["state"] = "normal"
        except Exception as e:
            messagebox.showerror("Error", f"Error importing data: {str(e)}")

    def preprocess_data(self):
        def dataProcessing(patient, labels_df, size=32, noslices=5):
            label = labels_df._get_value(patient, 'cancer')
            path = self.dataDirectory + patient
            slices = [dicom.dcmread(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            for each_slice in slices[:noslices]:
                img = np.array(each_slice.pixel_array)
                resized_img = cv2.resize(img, (size, size))
                new_slices.append(resized_img)

            img_data = np.array(new_slices)
            img_data = np.expand_dims(img_data, axis=-1)
            return img_data, np.array([0, 1]) if label == 1 else np.array([1, 0])

        try:
            imageData = []
            total_patients = len(self.lungPatients)
            
            for num, patient in enumerate(self.lungPatients):
                progress = (num + 1) / total_patients * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Processing patient {num + 1} of {total_patients}")
                self.root.update()

                try:
                    img_data, label = dataProcessing(patient, self.labels, size=self.size, noslices=self.NoSlices)
                    if img_data.shape == (self.NoSlices, self.size, self.size, 1):
                        imageData.append([img_data, label, patient])
                except KeyError:
                    print(f'Patient {patient} is unlabeled')

            imageData = np.array(imageData, dtype=object)
            np.save('imageDataNew-{}-{}-{}.npy'.format(self.size, self.size, self.NoSlices), imageData)
            
            self.progress_label.config(text="Pre-processing completed successfully!")
            messagebox.showinfo("Pre-Process Data", "Data Pre-Processing Done Successfully!")
            self.b2["state"] = "disabled"
            self.b3["state"] = "normal"
            
        except Exception as e:
            messagebox.showerror("Error", f"Error while processing data: {str(e)}")

    def train_data(self):
        try:
            imageData = np.load('imageDataNew-32-32-5.npy', allow_pickle=True)
            trainingData = imageData[:45]
            validationData = imageData[45:50]
            
            self.progress_var.set(0)
            self.progress_label.config(text="Initializing training...")
            
            model = Sequential([
                Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
                      padding='same', input_shape=(self.NoSlices, self.size, self.size, 1)),
                MaxPooling3D(pool_size=(1, 2, 2), padding='same'),
                Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
                MaxPooling3D(pool_size=(1, 2, 2), padding='same'),
                Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
                MaxPooling3D(pool_size=(1, 2, 2), padding='same'),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')
            ])

            model.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
            X_train = np.array([i[0] for i in trainingData])                       
            y_train = np.array([i[1] for i in trainingData])
            X_val = np.array([i[0] for i in validationData])
            y_val = np.array([i[1] for i in validationData])

            progress_callback = TrainingProgressCallback(
                self.progress_var,
                self.progress_label
            )

            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')

            try:
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=2,
                    validation_data=(X_val, y_val),
                    callbacks=[progress_callback]
                )

                sys.stdout = old_stdout

                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                self.progress_label.config(
                  text=f"Training completed! Final validation accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.4f}"
                )
                messagebox.showinfo(
                  "Train Data",
                 f"Model Trained Successfully!\nValidation Accuracy: {val_accuracy:.2f}\nValidation Loss: {val_loss:.4f}"
                  )

                
                model.save('lung_cancer_model.h5')
                
            except Exception as e:
                sys.stdout = old_stdout
                raise e

        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")

    def check_single_image(self):
        self.b4 = Button(text="Check Image", cursor="hand2", command=self.predict_cancer, 
                        font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b4.place(x=80, y=280, width=180, height=30)
    
    def predict_cancer(self):
        try:
            model = load_model('lung_cancer_model.h5')
            
            file_path = filedialog.askdirectory(title="Select Patient DICOM folder")
            if not file_path:
                return
                
            slices = [dicom.dcmread(file_path + '/' + s) for s in os.listdir(file_path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            
            new_slices = []
            for each_slice in slices[:self.NoSlices]:
                img = np.array(each_slice.pixel_array)
                resized_img = cv2.resize(img, (self.size, self.size))
                new_slices.append(resized_img)
                
            img_data = np.array(new_slices)
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.expand_dims(img_data, axis=0)
            
            prediction = model.predict(img_data, verbose=0)
            probability = prediction[0][1] * 100
            
            result_window = Toplevel(self.root)
            result_window.title("Detection Result")
            result_window.geometry("300x150")
            
            result_text = f"Cancer Probability: {probability:.2f}%\n"
            result_text += "High Risk" if probability > 50 else "Low Risk"
            
            result_label = Label(result_window, 
                               text=result_text,
                               font=("Times New Roman", 14),
                               pady=20)
            result_label.pack()
            
            ok_button = Button(result_window, 
                             text="OK",
                             command=result_window.destroy,
                             font=("Times New Roman", 12))
            ok_button.pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    root = Tk()
    obj = LCD_CNN(root)
    root.mainloop()