import numpy as np
import pandas as pd
import os.path
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping


def create_sepsis_dataset():
    with open('state_features.txt') as f:
        state_features = f.read().split()
    sepsis = pd.read_csv('mimictable.csv')

    
    binary_fields = ['gender','mechvent','re_admission']
    norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
    'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
    log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
              'input_total','input_4hourly','output_total','output_4hourly', 'bloc']

    # normalise binary fields
    sepsis.loc[:, binary_fields] = sepsis.loc[:, binary_fields] - 0.5 

    # normal distn fields
    for item in norm_fields:
        av = sepsis.loc[:, item].mean()
        std = sepsis.loc[:, item].std()
        sepsis.loc[:,item] = (sepsis.loc[:, item] - av) / std
        
    # log normal fields
    sepsis.loc[:, log_fields] = np.log(0.1 + sepsis.loc[:, log_fields])
    for item in norm_fields:
        av = sepsis.loc[:, item].mean()
        std = sepsis.loc[:, item].std()
        sepsis.loc[:,item] = (sepsis.loc[:, item] - av) / std

    # scale features to [0,1]
    for col in state_features:
        minimum = min(sepsis.loc[:,col])
        maximum = max(sepsis.loc[:,col])
        sepsis.loc[:,col] = (sepsis.loc[:,col] - minimum)/(maximum-minimum)
        
    # add rewards
    sepsis['reward'] = 0
    c0 = -0.1/4
    c1 = -0.5/4
    c2 = -2
    for i in sepsis.index:
        if i == 0:
            continue
        if sepsis.loc[i, 'icustayid'] == sepsis.loc[i-1, 'icustayid']:
            sofa_cur = sepsis.loc[i,'SOFA']
            sofa_prev = sepsis.loc[i-1,'SOFA']
            lact_cur = sepsis.loc[i,'Arterial_lactate']
            lact_prev = sepsis.loc[i-1,'Arterial_lactate']
            reward = 0
            if sofa_cur == sofa_prev and sofa_cur != 0:
                reward += c0
            reward += c1*(sofa_cur-sofa_prev)
            reward += c2*np.tanh(lact_cur - lact_prev)
            sepsis.loc[i-1,'reward'] = reward
        if i % 10000 == 0:
            print(i, flush = True)
    
    # create dataset of trajectories
    dataset=[]
    traj = []
    for i in sepsis.index:
        cur_state = sepsis.loc[i,state_features].to_numpy()
        if i == sepsis.index[-1]:
            # trajectory is finished
            traj.append(cur_state)
            dataset.append(traj)
            traj = []
        elif sepsis.loc[i, 'icustayid'] != sepsis.loc[i+1, 'icustayid']:    
            # trajectory is finished
            traj.append(cur_state)
            dataset.append(traj)
            traj = []
        else:
            action = np.array([sepsis.loc[i,'input_4hourly'],sepsis.loc[i, 'max_dose_vaso']])
            reward = sepsis.loc[i, 'reward']
            traj.extend([cur_state, action, reward])
        if i % 10000 == 0:
            print(i, flush = True)
    np.save('sepsis.npy', dataset)
    return dataset
    
def preprocess(dataset):
    X=[]
    y=[]
    for traj in dataset:
        for i in range(int(len(traj)/3)):
            # (state, action) pair
            X.append(np.append(traj[3*i], traj[3*i+1]))           
            # next state
            y.append(traj[3*i+3])
    
    x = np.asarray(X)
    y = np.asarray(y)
    return x, y

def build_envmodel(input_size, hidden_1_size, hidden_2_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_1_size, input_dim =input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(hidden_2_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_size))
    return model

def train_envmodel(train_path, input_size, hidden_1_size, hidden_2_size, output_size):
    train_set = np.load(train_path, allow_pickle=True) # need to change
    x_train, y_train = preprocess(train_set)
    # Callbacks
    checkpoint_path = 'env_best.hdf5'
    cp_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_best_only = True,
        monitor = 'val_loss',
        mode = 'min')
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

    EnvModel = build_envmodel(input_size, hidden_1_size, hidden_2_size, output_size)
    EnvModel.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae', 'mse'])
    EnvModel.summary()
    history = EnvModel.fit(x_train, 
                 y_train, 
                 validation_split = 0.2, 
                 batch_size=32, 
                 epochs=100,
                 callbacks = [early_stop, cp_callback],
                 verbose = 2)
    return EnvModel

if __name__ =="__main__":
    if not os.path.isfile('sepsis.npy'):
        create_sepsis_dataset()
    train_envmodel('sepsis.npy',50, 512, 512, 48)
    