import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Αντικατάσταση κόμματος με τελεία σε όλες τις στήλες
def convert_commas_to_periods(df):
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].str.replace(',', '.', regex=False)
    return df

# Φόρτωση του dataset
df_nm = pd.read_csv("/home/poulimenos/project/nm_features.csv")
df_pd = pd.read_csv("/home/poulimenos/project/pd_features.csv")
df_koa = pd.read_csv("/home/poulimenos/project/koa_features.csv")

# Αφαίρεση εγγραφών με ελλιπή δεδομένα
df_nm['Id'] = (
    df_nm['ID'].astype(str) + 
    df_nm['Disease'].astype(str)  + 
    df_nm['RIGHT_CLOSED_TO_CAMERA'].astype(str)
)
df_koa['Id'] = (
    df_koa['ID'].astype(str) + 
    df_koa['Disease'].astype(str) + 
    df_koa['Level'].astype(str) + 
    df_koa['RIGHT_CLOSED_TO_CAMERA'].astype(str)
)

df_pd['Id'] = (
    df_pd['ID'].astype(str) + 
    df_pd['Disease'].astype(str) + 
    df_pd['Level'].astype(str) + 
    df_pd['RIGHT_CLOSED_TO_CAMERA'].astype(str)
)
df_nm1=df_nm
df_koa1=df_koa
df_pd1=df_pd

df_nm1=df_nm1.dropna().drop(columns='Level', errors='ignore')
df_koa1=df_koa1.dropna().drop(columns='Level', errors='ignore')
df_pd1=df_pd1.dropna().drop(columns='Level', errors='ignore')

df_pd1 = convert_commas_to_periods(df_pd1)
df_koa1 = convert_commas_to_periods(df_koa1)
df_nm1 = convert_commas_to_periods(df_nm1)

label_encoder1 = LabelEncoder()
ds=pd.concat([df_nm1, df_koa1, df_pd1], ignore_index=True)
label_encoder1.fit(ds['Disease'])
# Κωδικοποίηση ετικετών
df_nm1['Disease'] = label_encoder1.transform(df_nm1['Disease'])
df_koa1['Disease'] = label_encoder1.transform(df_koa1['Disease'])
df_pd1['Disease'] = label_encoder1.transform(df_pd1['Disease'])

# Επιλογή των test δεδομένων με βάση το 'Id'
x_nm1_test = df_nm1[df_nm1['Id'] == "28NM1"]
x_koa1_test = df_koa1[df_koa1['Id'] == "5KOAMD0"]
x_pd1_test = df_pd1[df_pd1['Id'] == "1PDML1"]

y_nm1_test = x_nm1_test["Disease"]
y_koa1_test = x_koa1_test["Disease"]
y_pd1_test = x_pd1_test["Disease"] 

x_nm1_test=x_nm1_test.dropna().drop(columns=['Disease','Id'], errors='ignore')
x_koa1_test =x_koa1_test.dropna().drop(columns=['Disease','Id'], errors='ignore')
x_pd1_test=x_pd1_test.dropna().drop(columns=['Disease','Id'], errors='ignore')

# Δημιουργία των train δεδομένων (όλα τα υπόλοιπα εκτός των test)
x1_train = df_nm1[~df_nm1['Id'].isin(["28NM1","5KOAMD0","1PDML1"])]

# Δημιουργία των target τιμών για τα train σύνολα
y1_train = x1_train["Disease"]

# Αφαίρεση της στήλης "Disease_Level" από τα train δεδομένα
x1_train = x1_train.drop(columns=["Disease", "Id"], errors='ignore')

########################################
##
########################################
# Δημιουργία της στήλης Disease_Level
df_nm['Disease_Level'] = df_nm['Disease']
df_koa['Disease_Level'] = (
    df_koa['Disease'].astype(str) + '_' + df_koa['Level'].astype(str)
)
df_pd['Disease_Level'] = (
    df_pd['Disease'].astype(str) + '_' + df_pd['Level'].astype(str)
)

# Καθαρισμός δεδομένων
df_nm = df_nm.dropna().drop(columns=['ID', 'Disease', 'Level'], errors='ignore')
df_koa = df_koa.dropna().drop(columns=['ID', 'Disease', 'Level'], errors='ignore')
df_pd = df_pd.dropna().drop(columns=['ID', 'Disease', 'Level'], errors='ignore')

df_pd = convert_commas_to_periods(df_pd)
df_koa = convert_commas_to_periods(df_koa)
df_nm = convert_commas_to_periods(df_nm)

label_encoder = LabelEncoder()
ds1=pd.concat([df_nm, df_koa, df_pd], ignore_index=True)
label_encoder.fit(ds1['Disease_Level'])

# Κωδικοποίηση ετικετών
df_nm['Disease_Level'] = label_encoder.transform(df_nm['Disease_Level'])
df_koa['Disease_Level'] = label_encoder.transform(df_koa['Disease_Level'])
df_pd['Disease_Level'] = label_encoder.transform(df_pd['Disease_Level'])

# Επιλογή των test δεδομένων με βάση το 'Id'
x_nm_test1 = df_nm[df_nm['Id'] == "28NM1"]
x_koa_test1 = df_koa[df_koa['Id'] == "5KOAMD0"]
x_pd_test1 = df_pd[df_pd['Id'] == "1PDML1"]


x_koa_test2 = df_koa[df_koa['Id'] == "5KOASV0"]
x_pd_test2 = df_pd[df_pd['Id'] == "1PDMD1"]


x_koa_test3 = df_koa[df_koa['Id'] == "3KOAEL0"]
x_pd_test3 = df_pd[df_pd['Id'] == "1PDSV1"]

# Δημιουργία των target τιμών για τα test σύνολα
y_nm_test1 = x_nm_test1["Disease_Level"]
y_koa_test1 = x_koa_test1["Disease_Level"]
y_pd_test1 = x_pd_test1["Disease_Level"]
y_koa_test2 = x_koa_test2["Disease_Level"]
y_pd_test2 = x_pd_test2["Disease_Level"]
y_koa_test3 = x_koa_test3["Disease_Level"]
y_pd_test3 = x_pd_test3["Disease_Level"]

# Αφαίρεση της στήλης "Disease_Level" από τα test δεδομένα
x_nm_test1 = x_nm_test1.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_koa_test1 = x_koa_test1.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_pd_test1 = x_pd_test1.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_koa_test2 = x_koa_test2.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_pd_test2 = x_pd_test2.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_koa_test3 = x_koa_test3.drop(columns=["Disease_Level", "Id"], errors='ignore')
x_pd_test3 = x_pd_test3.drop(columns=["Disease_Level", "Id"], errors='ignore')

# Δημιουργία των train δεδομένων (όλα τα υπόλοιπα εκτός των test)
x_train = df_nm[~df_nm['Id'].isin(["28NM1", "5KOASV0","1PDMD1","3KOAEL0","1PDSV1","5KOAMD0","1PDML1"])]

# Δημιουργία των target τιμών για τα train σύνολα
y_train = x_train["Disease_Level"]

# Αφαίρεση της στήλης "Disease_Level" από τα train δεδομένα
x_train = x_train.drop(columns=["Disease_Level", "Id"], errors='ignore')


# Standard Scaler για τα δεδομένα (προαιρετικό, αν χρειάζεται κανονικοποίηση)
scaler = StandardScaler()

# Δημιουργία του μοντέλου RandomForest
clf = RandomForestClassifier(
    n_estimators=500, 
    max_depth=20, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    max_features='log2', 
    bootstrap=True
)

def evaluate_model(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    # Correct predictions count
   

    holdout_accuracy = accuracy_score(y_test, y_pred)
    holdout_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    holdout_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    holdout_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Hold-Out Validation Accuracy: {holdout_accuracy:.2f}")
    print(f"Hold-Out Precision: {holdout_precision:.2f}")
    print(f"Hold-Out Recall: {holdout_recall:.2f}")
    print(f"Hold-Out F1-Score: {holdout_f1:.2f}\n")
    
   # for i in range(y_test.shape[0]):
     #   print(f"predictions = {y_pred[i]} , True = {y_test[i]}")

# Data concatenation
x_test = pd.concat([x_nm_test1, x_koa_test1, x_pd_test1,x_koa_test2, x_pd_test2,x_koa_test3, x_pd_test3], ignore_index=True)
y_test = pd.concat([y_nm_test1, y_koa_test1, y_pd_test1, y_koa_test2, y_pd_test2, y_koa_test3, y_pd_test3], ignore_index=True)

print("LOSO Validation all LEVEL DISEASE as target:")
evaluate_model(clf, x_train, y_train, x_test, y_test)

##===================================
##
##===================================

# Data concatenation
x1_test = pd.concat([x_nm1_test, x_koa1_test, x_pd1_test], ignore_index=False)
y1_test = pd.concat([y_nm1_test, y_koa1_test, y_pd1_test], ignore_index=False)
print("LOSO Validation only  DISEASE as target:")
evaluate_model(clf, x1_train, y1_train, x1_test, y1_test)
