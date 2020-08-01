
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



root = Tk()
root.title('DETECTION OF FAILURE OF SATELLITE SENSOR USING DEEP LEARNING')
root.geometry('1050x900')
root.configure(background="#ccf381")

var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="#ccf381")
var.set("DETECTION OF FAILURE OF SATELLITE SENSOR USING DEEP LEARNING")
label.grid(row=0,columnspan=6)


timesteps = 0
size = 0
ptime = 0
user = ""
password = ""

def train_file():
     root1=Tk()
     root1.title("login page")
     root1.geometry('350x200')
     root1.configure(background="#ccf381")
     def login():
         user = E.get()
         password = E1.get()
         admin_login(user,password)
     L=Label(root1, text = "Username",bd=8,background="#ccf381",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 6,column=8)
     E=Entry(root1)
     E.grid(row = 6, column = 9)
     L1=Label(root1, text = "Password",bd=8,background="#ccf381",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 8,column=8)
     E1=Entry(root1,show="*")
     E1.grid(row = 8, column = 9)
     B1=Button(root1,text="Login",width=7,height=1,command=login,bd=8,background="#4831d4")
     B1.grid(row = 9, column =9)
     root1.mainloop()

def admin_login(user,password):
     #print(user,password)
     if user == "admin" and password == "admin":
         root3 = Tk()
         root3.title('choose file')
         root3.geometry('600x300')
         root3.configure(background="#ccf381")
         E2=Button(root3,text="Browse file",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="#4831d4",command=OpenFile_train)
         E2.place(x=200,y=100)
         
         root3.mainloop()  
     else:
         root3 = Tk()
         root3.title('ERROR')
         L2 = Label(root3, text = "The Username or Password is incorrect!!!",font=('arial',16,'bold'),fg='red',background="#ccf381").grid(row = 2)
         root3.mainloop()
         
def predict_file():
     root2=Tk()
     root2.title("predict")
     root2.geometry('800x500')
     root2.configure(background="purple1")
     
     B1=Button(root2, text = "Browse file",width=20,height=2,command=OpenFile_predict,bd=2,background="purple2",font=('arial',16,'bold'))
     B1.grid(row=0,column=0)
     
     """Entry_filename=Entry(root2,width=30)
     Entry_filename.grid(row=0,column=1)
     
     B2=Button(root2, text = "predict",width=20,height=2,command=predict,bd=2,background="purple2",font=('arial',16,'bold'))
     B2.grid(row=1,column=1)
     
     label = Label(root2,text="Accuracy ANN",width=20,height=2,bd=2,background="purple2",font=('arial',16,'bold'))
     label.grid(row=2,column=0)
     
     Entry_ann=Entry(root2)
     Entry_ann.grid(row=2,column=1) 
     
     label = Label(root2,text="Accuracy",width=20,height=2,bd=2,background="purple2",font=('arial',16,'bold'))
     label.grid(row=3,column=0) 
     
     Entry_1=Entry(root2)
     Entry_1.grid(row=3,column=1) 
    
     
     Entry_accuracy = Entry(root2)
     Entry_accuracy.grid(row=4,columnspan=2)"""
     
     root2.mainloop()
     

def OpenFile_train():
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("CSV File", "*.csv"),("All Files","*.*")),
                           title = "Choose a file.")
    try:
        with open(name,'r') as UseFile:
          train(name)
    except FileNotFoundError:
         print("No file exists")  
             
def OpenFile_predict():
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("CSV File", "*.csv"),("All Files","*.*")),title = "Choose a file.")
    try:
        with open(name,'r') as UseFile:
            global test_ip,test_op
            global test_pred
            global dataset_test
             
            dataset_test=pd.read_csv(name)
            test_ip=dataset_test.iloc[:,0:1].values
            test_op=dataset_test.iloc[:,1:2].values
             
            from sklearn.preprocessing import MinMaxScaler
            sc=MinMaxScaler(feature_range=(0,1))
            #Converts the test input to the same form as training input
            inputs=dataset_test['Input'].values
            inputs=inputs.reshape(-1,1)
            inputs=sc.fit_transform(inputs)
            
            global x_test
            x_test =[]
            global y_test
            y_test=[]
            global predicted_op
            predicted_op=[]
            
            global timesteps,size,ptime
            size=len(inputs)
            size=size-timesteps
            #creates a similar 2d array of 60 time steps for the test input
            for i in range(timesteps,size):
                x_test.append(inputs[i-timesteps:i,0])
                y_test.append(test_op[i+ptime,0])
            x_test=np.array(x_test)
            y_test=np.array(y_test)
            
            x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
            predicted_op= regressor.predict(x_test)
            test_pred=sc.inverse_transform(predicted_op)
            
             
            from PIL import ImageTk,Image
            plt.plot(test_ip,color='red', label='Real')
            plt.plot(test_pred,color='blue', label='Prediction')
            plt.title('DTG1 PREDICTION')
            plt.xlabel('Time')
            plt.ylabel('DTG1')
            plt.savefig('graph1.png',bbox_inches='tight')
            plt.show()
            image = Image.open("graph1.png")
            image = image.resize((250, 250), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image)  
            panel = Label(root, image=img)
            panel.image = img
            panel.grid(row=2,column=3)
        
             
            y_pred = classifier.predict(X_train2)
            y_pred = (y_pred > 0.5)
            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y, y_pred)
            am = accuracy_score(y, y_pred)
            print(am)
            print(cm)
            str1 ="Accuracy for training is : = {}\nConfusion matrix = {}".format(am,cm)
             
            sc3=StandardScaler()
            X_train4=sc3.fit_transform(test_pred)
            output=classifier.predict(X_train4)
            output=(output>0.5)
            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y_test, output)
            am = accuracy_score(y_test, output)
            print("Accuracy of the model is:", am*100,"%")
            print(cm)
            str2 = "\nAccuracy of the model is:{}%\nConfusion matrix  = {}".format(am*100,cm)
            
            data = str1+str2
            
            
            
            plt.plot(output,color='red',label='Prediction')
            plt.plot(test_op, color='blue', label='Real')
            plt.title('Sync Failure Prediction')
            plt.xlabel('Time')
            plt.ylabel('Failure')
            plt.legend()
            plt.savefig('graph2.png',dpi=199)
            plt.show()
                 
            #canvas = Canvas(graph2, width = 300, height = 300)      
            #canvas.pack()   
            image = Image.open("graph2.png")
            image = image.resize((250, 250), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image)  
            panel1 = Label(root, image=img)
            panel1.image = img
            panel1.grid(row=2,column=5)
            
            labelText = StringVar()
            labelText.set(data)
            output = Label(root, textvariable=labelText,width=45, height=6)
            output.grid(row=3,column=4)
            
            
    except FileNotFoundError:
             print("No file exists")     
    

x_train=[]
y_train=[]
test_pred = ""
X_train2 = []
def train(filename):
    alert = Tk()
    alert.title("alert")
    alert.geometry('250x50')
    label = Label(alert,text="Machine Trained",background="#c70039",font=('arial',16,'bold'))
    label.pack()
    
    global x_train,y_train,regressor,classifier,dataset_train,training_set,X_train2,X,y

    
    

    dataset_train=pd.read_csv(filename)
    training_set=dataset_train.iloc[:,0:1].values

    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(feature_range=(0,1))
    training_set_scaled=sc.fit_transform(training_set)
    global timesteps,ptime
    timesteps=60
    ptime=30
    
    tsize=len(training_set_scaled)
    tsize=tsize-timesteps
    for i in range(timesteps,tsize):
        x_train.append(training_set_scaled[i-timesteps:i,0])
        y_train.append(training_set_scaled[i+ptime,0])
    x_train,y_train=np.array(x_train), np.array(y_train) 
    x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))


    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    regressor=Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape= (x_train.shape[1],1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam',loss='mean_squared_error')

    regressor.fit(x_train,y_train,epochs=100, batch_size=32)


    #ANN
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 0:1].values
    y = dataset.iloc[:, 1:2].values

    sc2 = StandardScaler()
    X_train2 = sc2.fit_transform(X)
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'normal', activation = 'relu', input_dim = 1))
    classifier.add(Dense(units = 6, kernel_initializer = 'normal',activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train2, y, batch_size = 32, epochs = 100)
    
    
     
    
B = Button(root, text = "Train",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="#4831d4",command=train_file)
B.grid(row=1,column=0)

B1 = Button(root, text = "Predict",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="#4831d4",command=OpenFile_predict)
B1.grid(row=1,column=4)

root.mainloop()





