# Movie Conversational Chatbot using RNN Language Model 😼
This project implements a conversational chatbot for movie dialogs using a Recurrent Neural Network (RNN) language model. The chatbot is trained on a dataset of movie dialogs and utilizes TensorFlow 2.x for the model architecture and training.

# Too view the assignment for CS171, I suggest viewing it through the NLP_Chatbot.ipynb

# You can view the GPT Premade model as GPTPretrained_model.ipynb.

# Requirements
- Python 3.0 or better
- Packages: TensorFlow, NumPy, Pandas, Scikit-learn, flask  (You can install using pip install tensorflow, pip install numpy, pip install pandas, pip install scikit, pip install flask)
- Required to download from here: https://www.kaggle.com/datasets/hijest/cleaned-data-for-the-chatbot-collected-from-movies/code unzip and move all the files inside data folder
  
# Setup
- Open Terminal
- CD to Directory
- Download package using 'git clone https://github.com/realdarter/MovieChatBot-RNN'
- Using Python run the app.py

It will then begin training the model then show the application website in which you run it.
```
Epoch #1
Training Loss 2.04860520362854
Testing Loss 1.8013511896133423
##################################################
78/87 ━━━━━━━━━━━━━━━━━━━━ 8s 892ms/step2024-05-15 01:02:57.976736: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
86/87 ━━━━━━━━━━━━━━━━━━━━ 0s 827ms/step2024-05-15 01:02:59.526186: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Model is saved
##################################################
Epoch #2
Training Loss 1.7628453969955444
Testing Loss 1.772197961807251
##################################################
78/87 ━━━━━━━━━━━━━━━━━━━━ 7s 883ms/step2024-05-15 01:04:08.722391: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
86/87 ━━━━━━━━━━━━━━━━━━━━ 0s 817ms/step2024-05-15 01:04:10.170552: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Model is saved
##################################################
```
After the program has finished it will return a local server website  that you can access.



```
                   |\_ |\   
                   \` .. \      
              __,.-" =__Y=         
            ."        )                       
      _    /   ,    \/\_              #$#%  
     ((____|    )_-\ \_-`            @#$%*&
     `-----'`-----` `--`              #$%#   
```
