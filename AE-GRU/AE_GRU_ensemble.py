from keras.models import load_model

import  pandas as pd 
import numpy as np
df=pd.read_excel("Distributed_all.xlsx",sheet_name="Sayfa1")
date=pd.to_datetime("1st of January, 2015")
date_indexes=date+pd.to_timedelta(np.arange(1096*24),"H")

days=np.zeros((date_indexes.shape[0],1))
for i in range(date_indexes.shape[0]):
    if date_indexes[i].strftime("%A")=="Monday":
        days[i,0]=1
    elif date_indexes[i].strftime("%A")=="Tuesday":
        days[i,0]=2
    elif date_indexes[i].strftime("%A")=="Wednesday":
        days[i,0]=3
    elif date_indexes[i].strftime("%A")=="Thursday":
        days[i,0]=4
    elif date_indexes[i].strftime("%A")=="Friday":
        days[i,0]=5
    elif date_indexes[i].strftime("%A")=="Saturday":
        days[i,0]=6
    elif date_indexes[i].strftime("%A")=="Sunday":
        days[i,0]=7

#Hours coding
hours=np.zeros((26304,1))
y=np.array([i for i in range(1,25)])
for i in range(1096):
    hours[i*24:(i+1)*24,0]=y
    
temperature=df["Celcius"].get_values().reshape(26304,1)
loads=df["Tüketim Birimi"].get_values().reshape(26304,1)
data=pd.DataFrame({"dates":date_indexes})
data["days"]=days
data["hours"]=hours
data["temp"]=temperature
data["load"]=loads

training_first_point=data[data["dates"]=="2015-01-01 00:00:00"].iloc[0].name
training_last_point=data[data["dates"]=="2017-01-01 00:00:00"].iloc[0].name

dem=data["load"].get_values().astype(float).reshape(26304,1)
temp=data["temp"].get_values().astype(float).reshape(26304,1)
dayses=pd.get_dummies(data["days"]).get_values().astype(float)
hourses=data["hours"].get_values().astype(float).reshape(26304,1)
hourses=hourses-1
hourses=np.sin(2*np.pi*hourses/24)

#Training Part
training_inputs_dem=np.zeros(((training_last_point-168)*168,1))
training_inputs_temp=np.zeros(((training_last_point-168)*168,1))
training_inputs_days=np.zeros(((training_last_point-168)*168,7))
#training_inputs_hours=np.zeros(((training_last_point-168)*168,1))

for i in range(training_last_point-168):
    training_inputs_dem[i*168:(i+1)*168,0]=dem[i:i+168,0]
    training_inputs_temp[i*168:(i+1)*168,0]=temp[i+1:i+1+168,0]
    training_inputs_days[i*168:(i+1)*168,:]=dayses[i+1:i+1+168,:]
    #training_inputs_hours[i*168:(i+1)*168,0]=hourses[i+1:i+1+168,0]

#Reshaping    
training_inputs_dem=training_inputs_dem.reshape(-1,168,1)
training_inputs_temp=training_inputs_temp.reshape(-1,168,1)
training_inputs_days=training_inputs_days.reshape(-1,168,7)
#training_inputs_hours=training_inputs_hours.reshape(-1,168,1)
#training_inputs_temp=np.dstack([training_inputs_temp,training_inputs_hours])

#Normalizations
train_mean_loads=np.mean(dem[:training_last_point])
train_std_loads=np.std(dem[:training_last_point])

train_mean_temps=np.mean(temp[:training_last_point])
train_std_temps=np.std(temp[:training_last_point])

training_inputs_dem[:,:,0]=(training_inputs_dem[:,:,0]-train_mean_loads)/train_std_loads
training_inputs_temp[:,:,0]=(training_inputs_temp[:,:,0]-train_mean_temps)/train_std_temps

training_inputs_dt=np.dstack([training_inputs_dem,training_inputs_temp,training_inputs_days])

#Train labels
train_labs=dem[168:training_last_point].reshape((training_last_point-168),1)
train_labs=(train_labs-train_mean_loads)/train_std_loads

#TEST PART
testing_inputs_dem=np.zeros(((len(loads)-training_last_point)*168,1))
testing_inputs_temp=np.zeros(((len(loads)-training_last_point)*168,1))
testing_inputs_days=np.zeros(((len(loads)-training_last_point)*168,7))
#testing_inputs_hours=np.zeros(((len(loads)-training_last_point)*168,1))

for i in range(len(loads)-training_last_point):
    testing_inputs_dem[i*168:(i+1)*168,0]=dem[(training_last_point-168)+i:training_last_point+i,0]
    testing_inputs_temp[i*168:(i+1)*168,0]=temp[(training_last_point-168)+i+1:training_last_point+i+1,0]
    testing_inputs_days[i*168:(i+1)*168,:]=dayses[(training_last_point-168)+i+1:training_last_point+i+1,:]
    #testing_inputs_hours[i*168:(i+1)*168,0]=hourses[(training_last_point-168)+i+1:training_last_point+i+1,0]
    

testing_inputs_dem=testing_inputs_dem.reshape(-1,168,1)
testing_inputs_temp=testing_inputs_temp.reshape(-1,168,1)
testing_inputs_days=testing_inputs_days.reshape(-1,168,7)
#testing_inputs_hours=testing_inputs_hours.reshape(-1,168,1)
#testing_inputs_temp=np.dstack([testing_inputs_temp,testing_inputs_hours]) 

#Normalizations
test_mean_loads=np.mean(dem[training_last_point:])
test_std_loads=np.std(dem[training_last_point:])

test_mean_temps=np.mean(temp[training_last_point:])
test_std_temps=np.std(temp[training_last_point:])

testing_inputs_dem[:,:,0]=(testing_inputs_dem[:,:,0]-test_mean_loads)/test_std_loads
testing_inputs_temp[:,:,0]=(testing_inputs_temp[:,:,0]-test_mean_temps)/test_std_temps

testing_inputs_dt=np.dstack([testing_inputs_dem,testing_inputs_temp,testing_inputs_days])

#Test labels
test_labs=dem[training_last_point:].reshape((len(loads)-training_last_point),1)
test_labs=(test_labs-test_mean_loads)/test_std_loads


shape_1=168
shape_2=9
b_size=24

#Loading Models

def load_all_models(n_models):
    all_models=[]
    for i in range(n_models):
        #define filename for this ensemble
        filename="AE_GRU_model_" + str(i+1) + ".h5"
        #load model from file
        model=load_model(filename)
        #add to list of members
        all_models.append(model)
        print(">loaded %s" %filename)
    return all_models



def ensemble_predictions(members,in_dem,test_labels):
    total_mapes=[]
    total_preds=[]
    total_results=[]
    for k in range(10):
        pred_model=members[k]
        mape=[]
        preds=[]
        for j in range(0,8737,24):
            x_input=in_dem[j:j+24,:,:]
            #t_inputs_days=testing_inputs_days[j:j+24,:,:]
            labels=test_labels[j:j+24,0]
            yhat = np.zeros(24)
            for i in range(23):
                yhat[i]=pred_model.predict([x_input[i,:,:].reshape(1,shape_1,shape_2)],batch_size=b_size)
                x_input[i+1,shape_1-1-i:shape_1,0] = yhat[:i+1] 
        
            yhat[i+1]=pred_model.predict([x_input[i+1,:,:].reshape(1,shape_1,shape_2)],batch_size=b_size)
            
            yhat=(yhat*test_std_loads)+test_mean_loads
            preds.append(yhat)
            labels=(labels*test_std_loads)+test_mean_loads
            perf=np.mean(np.abs(yhat-labels)/labels)
            mape.append(perf)
        
        c=[0,31,28,31,30,31,30,31,31,30,31,30,31]
        d=np.cumsum(c)
        
        last=[]
        for m in range(12):
            last.append(np.mean(mape[d[m]:d[m+1]]))
        results=[round(elem,4) for elem in last]
        
        #Results merging
        total_preds.append(preds)
        total_results.append(results)
        total_mapes.append(mape)
    return total_results,total_preds
    
    
    
    
    
members=load_all_models(10)    
ten_results,ten_preds=ensemble_predictions(members,testing_inputs_dt,test_labs)
 
months_ensemble=[]        
for p in range(12):
    month=0
    for q in range(10):
        month+=ten_results[q][p]
    month=month/10
    months_ensemble.append(month)
print("Ensembles for each month: ",np.round(months_ensemble,4))
print("MAPE result in total: ",np.round(np.mean(months_ensemble),5))

AE_GRU_preds=np.zeros((8760,10))
AE_GRU_hours_preds=np.zeros((8760,1))
for i in range(10):
    y=np.asarray(ten_preds[i])
    y=y.reshape(8760)
    AE_GRU_preds[:,i]=y

for i in range(8760):
    AE_GRU_hours_preds[i,0]=np.mean(AE_GRU_preds[i,:])

import scipy.io
p_dict={"AE_GRU_hours_preds": AE_GRU_hours_preds}
scipy.io.savemat("AE_GRU_h_preds",p_dict)



"""
s_1=np.zeros((8760,1))        
s_2=np.zeros((8760,1))        
s_3=np.zeros((8760,1))        
s_4=np.zeros((8760,1))        
s_5=np.zeros((8760,1))        
s_6=np.zeros((8760,1))        
s_7=np.zeros((8760,1))        
s_8=np.zeros((8760,1))        
s_9=np.zeros((8760,1))        
s_10=np.zeros((8760,1))        

for i in range(10):
    candidate=total_preds[i]
    for j in range(365):
        s_+str(i+1)[j*24:(j+1)*24,0]=candidate[j]        

preds_hours=(s_1+s_2+s_3+s_4+s_5+s_6+s_7+s_8+s_9+s_10)/10
"""     
        