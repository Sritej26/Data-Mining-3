
import numpy as np
import pandas as pd


from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances


insulin_data = pd.read_csv('InsulinData.csv',parse_dates=[['Date','Time']])
isn = insulin_data[['Date_Time', 'BWZ Carb Input (grams)']]
isn = isn.rename(columns={'BWZ Carb Input (grams)': 'meal'})

cgm_data = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']], keep_date_col=True)
cgm = cgm_data[['Date_Time','Index','Sensor Glucose (mg/dL)','ISIG Value','Date','Time']]
cgm = cgm.rename(columns={'Sensor Glucose (mg/dL)':'sensor','ISIG Value':'isg','Index':'Index'})
cgm['Index'] = cgm.index



def gl_d(cgm, insulin):
    imt = mo_g(insulin)
    res = merging(cgm, imt)
    return res

def merging(cdp, imp):
    dataFrame2 = cdp.copy()
    dataFrame1 = imp.copy()
    dataFrame2 = dataFrame2.loc[dataFrame2['sensor'].notna()]
    dataFrame2.set_index(['Date_Time'],inplace=True)
    dataFrame2 = dataFrame2.sort_index()
    dataFrame2 = dataFrame2.reset_index()
    dataFrame1.set_index(["Meal_Time"],inplace=True)
    dataFrame1 = dataFrame1.sort_index()
    dataFrame1 = dataFrame1.reset_index()
    res = pd.merge_asof(dataFrame1, dataFrame2,left_on='Meal_Time',right_on='Date_Time',direction="forward")
    return res

def mo_g(dataFrame1):
    isn = dataFrame1.copy()
    isn = isn.loc[isn['meal'].notna()&isn['meal'] != 0]
    isn.set_index(['Date_Time'],inplace=True)
    isn = isn.sort_index()
    isn = isn.reset_index()
    isnd = isn.diff(axis=0)
    isnd = isnd.loc[isnd['Date_Time'].dt.seconds >= 7200]
    isn = isn.join(isnd,lsuffix='_caller',rsuffix='_other')
    isn = isn.loc[isn['Date_Time_other'].notna(),['Date_Time_caller','meal_caller']]
    isn = isn.rename(columns={'Date_Time_caller':'Meal_Time','meal_caller':'meal'})
    return isn



def sen(df, vc):
    dc = df.loc[df['sensor'].notna()]['sensor'].count()
    if dc < vc:
        return False, None
    dtBefore = None
    val = 0
    for x in df.iterrows():
        if dtBefore == None:
            dtBefore = x[1]['Date_Time']
            val += 1
            continue
        if (x[1]['Date_Time'] - dtBefore).seconds < 300:
            df.at[val, 'sensor'] = -999
            val += 1
            continue
        dtBefore = x[1]['Date_Time']
        val += 1
    df = df.loc[df['sensor'] != -999]
    if df['sensor'].count() == vc:
        return True, df
    else:
        return False, None



def fm(dataFrameMeal, dataFrame2):
    inputm = []
    dataFrameMeal.reset_index()

    for index, x in dataFrameMeal.iterrows():
        begin = x['Date_Time'] + pd.DateOffset(minutes=-30)
        stop = x['Date_Time'] + pd.DateOffset(hours=2)
        meal = dataFrame2.loc[(dataFrame2['Date_Time'] >= begin)&(dataFrame2['Date_Time']<stop)]
        meal.set_index('Date_Time',inplace=True)
        meal = meal.sort_index()
        meal = meal.reset_index()
        isCorrect, meal = sen(meal, 30)
        if isCorrect == False:
            continue
        meal = meal[['sensor']]
        arr = meal.to_numpy().reshape(1, 30)
        arr = np.insert(arr, 0, index, axis=1)
        arr = np.insert(arr, 1, x['meal'], axis=1)
        inputm.append(arr)

    return np.array(inputm).squeeze()


def get_feature(input):
    dataFrame = pd.DataFrame(data=input)
    df = pd.DataFrame(data=dataFrame.min(axis=1), columns=['min'])
    df['max'] = dataFrame.max(axis=1)
    df['sum'] = dataFrame.sum(axis=1)
    df['median'] = dataFrame.median(axis=1)
    df['min_max'] = df['max']-df['min']
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)


meal_data = gl_d(cgm, isn)
inputm = fm(meal_data, cgm)
scaler = MinMaxScaler()
inputm[:, 1:2]
requiredInput = get_feature(inputm[:, 1:2])
digitInput = scaler.fit_transform(inputm[:, 1:2])
min = digitInput.min()
max = digitInput.max()


transformFitData = scaler.fit_transform([[5],[26],[46],[66],[86],[106],[126]])
digitData = np.digitize(digitInput.squeeze(), transformFitData.squeeze(), right=True)

def cal_Entropy(labels, transformFitData):
    entropy = 0
    for label in np.unique(labels):
        lblPts = np.where(labels == label)
        localEntropy = 0
        count = 0
        unique, count = np.unique(transformFitData[lblPts], return_counts=True)
        for index in range(0, unique.shape[0]):
            exp = count[index] / float(len(lblPts[0]))
            localEntropy += -1*exp*np.log(exp)
        entropy += localEntropy * (len(lblPts[0]) / float(len(labels)))
    return entropy


def cal_Purity(labels, transformFitData):
    purity = 0
    for label in np.unique(labels):
        lblPts = np.where(labels == label)
        localPurity = 0
        count = 0
        unique, count = np.unique(transformFitData[lblPts], return_counts=True)
        for index in range(0, unique.shape[0]):
            exp = count[index] / float(len(lblPts[0]))
            if exp > localPurity:
                localPurity = exp

        purity += localPurity * (len(lblPts[0]) / float(len(labels)))
    return purity

kmeans_kwargs = {
    "init":"random",
    "n_init":10,
    "max_iter":100,
    "random_state":0
}

kmeans = KMeans(n_clusters=6, **kmeans_kwargs)
lblPredicates = kmeans.fit_predict(requiredInput)
temp_ind = digitData + 1
kEntropy = cal_Entropy(lblPredicates, temp_ind)
count=1.88
kPurity = cal_Purity(lblPredicates, temp_ind)
dbScan = DBSCAN(eps=0.03, min_samples=6).fit(requiredInput)
labels = dbScan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
dbScan.core_sample_indices_
dist = []
sse = 0

for l in np.unique(dbScan.labels_):
    if l == -1:
        continue
    center_points = np.where(dbScan.labels_ == l)
    center = np.mean(requiredInput[center_points], axis=0)
    sse += np.sum(np.square(euclidean_distances([center], requiredInput[center_points])), axis=1)
temp_ind = digitData+1
db_entropy = cal_Entropy(dbScan.labels_, temp_ind)
db_purity = count * cal_Purity(dbScan.labels_, temp_ind)
 

results = np.array([[kmeans.inertia_,sse,kEntropy,db_entropy,kPurity,db_purity]])
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
