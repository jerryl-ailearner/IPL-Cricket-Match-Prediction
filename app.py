import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#import xgboost
import tensorflow as tf 
import pandas as pd

app = Flask(__name__)

#model_LR_0_5_loaded_model = pickle.load(open('model_LR_0_5.pkl','rb'))
#model_LR_6_10_loaded_model = pickle.load(open('model_LR_6_10.pkl', 'rb'))
#model_LR_11_15_loaded_model = pickle.load(open('model_LR_11_15.pkl', 'rb'))
#model_LR_16_20_loaded_model = pickle.load(open('model_LR_16_20.pkl', 'rb'))
#model_LR_1_20_loaded_model = pickle.load(open('model_LR_1_20.pkl', 'rb'))

########################### 0 - 5 ##################################################
SVC_0_5_loaded_model = pickle.load(open('model_SVC_0_5.pkl', 'rb'))
#LR_0_5_loaded_model = pickle.load(open('model_LR_0_5.pkl', 'rb'))

######################### 6- 10  ###############################################
SVC_6_10_loaded_model = pickle.load(open('model_SVC_6_10.pkl', 'rb'))


############################# 11 -15 ##########################################
SVC_11_15_loaded_model = pickle.load(open('model_SVC_11_15.pkl', 'rb'))
LR_11_15_loaded_model = pickle.load(open('model_LR_11_15.pkl', 'rb'))

##############################  16 - 20  ###########################################
o2o_MF_stacked_lstm_16_20_loaded_model = tf.keras.models.load_model('model_o2o_MF_stacked_lstm_16_20.h5')

@app.route('/')
def home():
    return render_template('index.html')

mm_scaler = MinMaxScaler()
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    if request.method == 'POST':
    
        print('###############################  Data Entered ##########################################################\n' )        
        over =  int(request.form['over'] )  
        print('over-',over)          
        ball =  int(request.form['ball'])
        print('ball-',ball)  
        
       
        
        batsman =  str(request.form['batsman'])
        print('batsman-',batsman)  
        batman_score =  int(request.form['batman_score'])
        print('batman_score - ',batman_score)


        bowler =  str(request.form['bowler'])
        print('bowler-',bowler)
        bowler_run =  int(request.form['batman_score']) #int(request.form['bowler_run'])        
        print('bowler_run-',bowler_run)
        
        extra_runs =  int(request.form['extra_runs'] )
        print('extra_runs-',extra_runs)
        #batsman_runs =  int(request.form['batsman_runs'] )
        batsman_runs = int(batman_score) + int(extra_runs) 
        print('batsman_runs-',batsman_runs)


        present_score =  int(request.form['present_score'] )
        print('present_score-',present_score)
        target =  int(request.form['target'] )
        print('target-',target)
        wkts_left =  int(request.form['wkts_left'] )
        print('wkts_left-',wkts_left)
        
        #metric =  str(request.form['metric'] )
        metric = 'accuracy'

        print('target-',target)
        
        #balls_played =  int(request.form['balls_played'])
        a = np.dot(int(over-1),6)
        balls_played = np.add(a,int(ball))
        print('balls_played-',balls_played) 
        
        #remaining_balls =  int(request.form['remaining_balls'] )
        remaining_balls = 120 - balls_played
        print('remaining_balls-',remaining_balls)
        
        
        print('#################################################################################################################\n' )
        
     
                   
        dic_bat = {'Batsman1':0,'Batsman2':0,'Batsman3':0,'Batsman4':0,'Batsman5':0,'Batsman6':0,'Batsman7':0,'Batsman8':0,'Batsman9':0,\
           'Batsman10':0,'Batsman11':0}
        dic_bowl = {'bowler1':0,'bowler2':0,'bowler3':0,'bowler4':0,'bowler5':0,'bowler6':0,'bowler7':0,'bowler8':0,'bowler9':0,\
                    'bowler10':0} 


        string1 = batsman
        batsman_no = int(string1.strip('Batsman'))
        print('Batsman'+str(batsman_no),"- ",batman_score)        
        dic_bat['Batsman'+str(batsman_no)]  = batman_score
        print(dic_bat['Batsman'+str(batsman_no)])

        string2 = bowler
        Bowler_no = int(string2.strip('Bowler'))
        print('Bowler'+str(Bowler_no),"- ",bowler_run)        
        dic_bowl['bowler'+str(Bowler_no)]  = bowler_run
        print(dic_bowl['bowler'+str(Bowler_no)])
        

                                                                                #if(batsman=='Batsman1'): 
                                                                                    #print('Batsman1-',batman_score)        
                                                                                    #dic_bat['Batsman1']  = batman_score


                                                      ######################################################################

                                                                                    #if(bowler=='Bowler1'): 
                                                                                        #print('Bowler1',bowler_run)      
                                                                                        #dic_bowl['Bowler1']  = bowler_run           

     
        #print('###############################  Data Mapping ##########################################################' )
        
        bman1=dic_bat['Batsman1']
        bman2=dic_bat['Batsman2']
        bman3=dic_bat['Batsman3']
        bman4=dic_bat['Batsman4']
        bman5=dic_bat['Batsman5']
        bman6=dic_bat['Batsman6']
        bman7=dic_bat['Batsman7']
        bman8=dic_bat['Batsman8']
        bman9=dic_bat['Batsman9']
        bman10=dic_bat['Batsman10']
        bman11=dic_bat['Batsman11']
        
        bowl1=dic_bowl['bowler1']
        bowl2=dic_bowl['bowler2']
        bowl3=dic_bowl['bowler3']
        bowl4=dic_bowl['bowler4']
        bowl5=dic_bowl['bowler5']
        bowl6=dic_bowl['bowler6']
        bowl7=dic_bowl['bowler7']
        bowl8=dic_bowl['bowler8']
        bowl9=dic_bowl['bowler9']
        bowl10=dic_bowl['bowler10']

        #print('###############################  Data Input ##########################################################' )
        
        feature_values= [over,ball,bman1,bman2,bman3,bman4,bman5,bman6,bman7,\
                                          bman8,bman9,bman10,bman11,bowl1,bowl2,bowl3,bowl4,bowl5,bowl6,bowl7,bowl8,\
                                          bowl9,bowl10,extra_runs,batsman_runs,balls_played,remaining_balls,wkts_left,present_score,target]
        
        feature_values_dic = {'over':over,'ball':ball,'bman1':bman1,'bman2':bman2,'bman3':bman3,'bman4':bman4,'bman5':bman5,'bman6':bman6,'bman7':bman7,\
                                          'bman8':bman8,'bman9':bman9,'bman10':bman10,'bman11':bman11,'bowl1':bowl1,'bowl2':bowl2,'bowl3':bowl3,'bowl4':bowl4,\
                                           'bowl5':bowl5,'bowl6':bowl6,'bowl7':bowl7,'bowl8':bowl8,'bowl9':bowl9,'bowl10':bowl10,'extra_runs':extra_runs,\
                  'batsman_runs':batsman_runs,'balls_played':balls_played,'remaining_balls':remaining_balls,'wkts_left':wkts_left,'present_score':present_score,'target':target}

        feature_values_df = pd.DataFrame(feature_values_dic.values(),feature_values_dic.keys())
        feature_values_df_t = feature_values_df.transpose()

        feature_values_df_t.iloc[:,0:16]

        feature_values_df_t.iloc[:,16:]


        ipldata_sample_X = np.array(feature_values_df_t)

        print("********************** reshape for LSTM and GRU *************************************************\n")

        ipldata_sample_X_deep = np.reshape(ipldata_sample_X, (ipldata_sample_X.shape[0], 1, ipldata_sample_X.shape[1]))

        print(ipldata_sample_X.shape,ipldata_sample_X_deep.shape)

        print("ipldata_sample_X - \n",ipldata_sample_X)


        print("ipldata_sample_X_deep - \n",ipldata_sample_X_deep)
        
        #print('###############################  Deep ##########################################################' )
        
       
        
        
        
        #print('###############################  scaling ##########################################################' )
        
        
        #mm_scaler = StandardScaler()
        #ipldata_sample_X = mm_scaler.fit_transform(ipldata_sample_X)

        #print(ipldata_sample_X )
        
        score_0_5_acc = 63.87 # 68.68
        score_6_10_acc = 64.46 # 72.36
        score_11_15_acc = 72.66 # 79.22
        score_16_20_acc = 83.82 # 77.93
        
        score_0_5_pre = 79.27 # SVC
        score_6_10_pre =  75.95 #svc
        score_11_15_pre = 90.26 #SVC 
        score_16_20_pre =  73.99 #SVC     73.65 # RF    
     
        score_0_5_rec =   69.39  #SVC
        score_6_10_rec =  81.87  #SVC
        score_11_15_rec = 83.48 # SVC
        score_16_20_rec =  100.00 # LR 

        score_0_5_f1 =  73.46 # SVC
        score_6_10_f1 =  78.80 # SVC
        score_11_15_f1 = 83.41 # SVC
        score_16_20_f1 = 84.08 # LR
        
        print('###############################  Prediction ##########################################################' )
        
        if((metric == 'accuracy') and over<=5):
           prediction=SVC_0_5_loaded_model.predict(ipldata_sample_X)
           score = score_0_5_acc
           print('model for over = 1- 5 is executed ') 
        elif((metric == 'accuracy') and (over>= 6) and (over<= 10)):
          prediction=SVC_6_10_loaded_model.predict(ipldata_sample_X)
          score = score_6_10_acc
          print('model for over = 6- 10 is executed ') 
        elif((metric == 'accuracy') and (over>= 11) and (over<= 15)):
          prediction=SVC_11_15_loaded_model.predict(ipldata_sample_X)
          score = score_11_15_acc
          print('model for over = 11- 15 is executed ')   
        elif((metric == 'accuracy') and (over>= 16) ):
          print("o2o_MF_stacked_lstm_16_20_loaded_model")
          prediction=np.argmax(o2o_MF_stacked_lstm_16_20_loaded_model.predict(ipldata_sample_X_deep),axis=1)
          score = score_16_20_acc
          print('model for over = 16- 20 is executed ') 





        if((metric == 'precision') and over<=5):
           prediction=LR_0_5_loaded_model.predict(ipldata_sample_X)
           score = score_0_5_pre
           print('model for over = 1- 5 is executed ') 
        elif((metric == 'precision') and (over>= 6) and (over<= 10)):
          prediction=SVC_6_10_loaded_model.predict(ipldata_sample_X)
          score = score_6_10_pre
          print('model for over = 6- 10 is executed ') 
        elif((metric == 'precision') and (over>= 11) and (over<= 15)):
          prediction=RF_11_15_loaded_model.predict(ipldata_sample_X)
          score = score_11_15_pre
          print('model for over = 11- 15 is executed ')   
        elif((metric == 'precision') and (over>= 16) ):
          prediction=XGB_16_20_loaded_model.predict(ipldata_sample_X) 
          score = score_16_20_pre
          print('model for over = 16- 20 is executed ') 




        if((metric == 'recall') and over<=5):
           prediction=SVC_0_5_loaded_model.predict(ipldata_sample_X)
           score = score_0_5_rec
           print('model for over = 1- 5 is executed ') 
        elif((metric == 'recall') and (over>= 6) and (over<= 10)):
          prediction=SVC_6_10_loaded_model.predict(ipldata_sample_X)
          score = score_6_10_rec
          print('model for over = 6- 10 is executed ') 
        elif((metric == 'recall') and (over>= 11) and (over<= 15)):
          prediction=SVC_11_15_loaded_model.predict(ipldata_sample_X)
          score = score_11_15_rec
          print('model for over = 11- 15 is executed ')   
        elif((metric == 'recall') and (over>= 16) ):
          prediction=LR_16_20_loaded_model.predict(ipldata_sample_X) 
          score = score_16_20_rec
          print('model for over = 16- 20 is executed ') 




        if((metric == 'f1_score') and over<=5):
           prediction=SVC_0_5_loaded_model.predict(ipldata_sample_X)
           score = score_0_5_f1
           print('model for over = 1- 5 is executed ') 
        elif((metric == 'f1_score') and (over>= 6) and (over<= 10)):
          prediction=SVC_6_10_loaded_model.predict(ipldata_sample_X)
          score = score_6_10_f1
          print('model for over = 6- 10 is executed ') 
        elif((metric == 'f1_score') and (over>= 11) and (over<= 15)):
          prediction=SVC_11_15_loaded_model.predict(ipldata_sample_X)
          score = score_11_15_f1
          print('model for over = 11- 15 is executed ')   
        elif((metric == 'f1_score') and (over>= 16) ):
          prediction=LR_16_20_loaded_model.predict(ipldata_sample_X) 
          score = score_16_20_f1
          print('model for over = 16- 20 is executed ') 
          
        print('prediction \n',prediction)

        if(prediction[0]==0):
           output='win' #round(prediction[0],2)
        else:
           output='loose'
        print('\n\n******************************   PREDICTION -   Batting team is going to {}           ****************************************\n\n'.format(output))
        
        return render_template('index.html', prediction_text='Batting team will have chance to {} this match by {}%'.format(output,score))
        
        #return render_template('index.html', request.form['over']=str(over))
        #return render_template('index.html', request.form['ball']=str(ball))
        #return render_template('index.html', request.form['batsman']=str(batsman))
        #return render_template('index.html', request.form['batman_score']=str(batman_score))
        #return render_template('index.html', request.form['bowler']=str(bowler))
        #return render_template('index.html', request.form['extra_runs']=str(extra_runs))    
        #return render_template('index.html', request.form['present_score']=str(present_score))
        #return render_template('index.html', request.form['target']=str(target))
        #return render_template('index.html', request.form['wkts_left']=str(wkts_left))        
        
        
    else:
        return render_template('index.html')
        
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
 
