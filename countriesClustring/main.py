import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import json

def get_reg(y):
    """ 
        we consider X is [x1, x2, x3, ..., xn], also consider y is [y1, y2, y3, ..., yn]
        for using in the sklearn.LinearRegression X must be [[x1],
                                                             [x2],
                                                             [x3],
                                                                .
                                                                .
                                                                .
                                                             [xn]]
        at the end we get intercept and coefficent.
        
            Return [       reg, 
                            X,
                    intercept,
                    coefficent
                ]
    """

    reg = linear_model.LinearRegression()
    X = np.arange(len(y)).reshape(-1,1)
    y = y.reshape(-1,1)
    reg.fit(X, y)

    return [reg.intercept_[0], reg.coef_[0][0], reg.score(X,y)]
    
# Load and transform data --------

df_import = pd.read_csv('data/API_BM.GSR.GNFS.CD_DS2_en_csv_v2_6304000.csv')
df_import.drop(columns=df_import.columns[-1], axis=1, inplace=True)

df_export = pd.read_csv('data/API_NE.EXP.GNFS.CD_DS2_en_csv_v2_6300789.csv')
df_export.drop(columns=df_export.columns[-1], axis=1, inplace=True)

df_gdp = pd.read_csv('data/GDP_filled_miss_values_knn.csv')

df_gdppc = pd.read_csv('data/GDP_pcap_filled_miss_values_knn.csv')
df_gdppc.drop(columns=['Unnamed: 67'], axis=1, inplace=True)

df_oilp = pd.read_csv('data/API_NY.GDP.PETR.RT.ZS_DS2_en_csv_v2_6305850.csv')
df_oilp.drop(columns=df_oilp.columns[-1], axis=1, inplace=True)

df_gini = pd.read_csv('data/API_SI.POV.GINI_DS2_en_csv_v2_6508497.csv')
df_gini.drop(columns=df_gini.columns[-1], axis=1, inplace=True)

df_deathr = pd.read_csv('data/API_SP.DYN.CDRT.IN_DS2_en_csv_v2_6303594.csv')
df_deathr.drop(columns=df_deathr.columns[-1], axis=1, inplace=True)

# --------------------------------


# --------------------------------
# Analysis oil
all_years = [str(year) for year in range(1960,2023)]
years = [str(year) for year in range(2009, 2022)]
df_oilp.fillna(0, inplace=True)
col_rem_oilp = df_oilp.columns
df_oilp['avg oil gdp'] = df_oilp[years].sum(axis=1)/len(years)
df_oilp.drop(labels=col_rem_oilp, axis=1, inplace=True)
temp1 = df_oilp.copy()
print(temp1)

# --------------------------------
# Analysis GDPs
col_rem_gdp = df_gdp.columns
mt_gdp_log = np.log(df_gdp[all_years].to_numpy())

df_gdp['log_gdp'] = 0.0
df_gdp['inter_reg_gdp'] = 0.0
df_gdp['beta_reg_gdp'] = 0.0
df_gdp['r2_reg_gdp'] = 0.0

for j in range(len(mt_gdp_log)):
    reg = get_reg(mt_gdp_log[j])
    df_gdp.at[j, 'log_gdp'] = mt_gdp_log[j][-1]
    df_gdp.at[j, 'inter_reg_gdp'] = reg[0]
    df_gdp.at[j, 'beta_reg_gdp'] = reg[1]
    df_gdp.at[j, 'r2_reg_gdp'] = reg[2]

df_gdp.drop(labels=col_rem_gdp, axis=1, inplace=True)
temp2 = pd.concat([df_gdp,temp1], axis=1)
print(temp2)


# ---------------------------------
# Analysis GDP per capita
mt_gdp_percap_log = np.log(df_gdppc[all_years].to_numpy())

df_gdppc['log_gdp_percap'] = 0.0
df_gdppc['inter_gdp_percap'] = 0.0
df_gdppc['beta_gdp_percap'] = 0.0
df_gdppc['r2_gdp_percap'] = 0.0

for j in range(len(mt_gdp_percap_log)):
    reg = get_reg(mt_gdp_percap_log[j])
    df_gdppc.at[j, 'log_gdp_percap'] = mt_gdp_percap_log[j][-1]
    df_gdppc.at[j, 'inter_gdp_percap'] = reg[0]
    df_gdppc.at[j, 'beta_gdp_percap'] = reg[1]
    df_gdppc.at[j, 'r2_gdp_percap'] = reg[2]

df_gdppc.drop(labels=all_years, axis=1, inplace=True)
temp3 = pd.concat([df_gdppc,temp2], axis=1)


# ---------------------------------

temp3 = temp3.drop(['Indicator Code', 'Indicator Name', 'Country Code'], axis=1)

cols = temp3.columns
cols = cols.drop(['Country Name'])
print(temp3[cols])


print(temp3[cols].corr())
sns.heatmap(temp3[cols].corr(), annot=True)
plt.show()

# ---------------------------------
# Beacuse of correlation between log_gdp and inter_reg_gdp
# we remove one of them (inter_reg_gdp)

temp3.drop(['inter_reg_gdp'], axis=1, inplace=True)
print(temp3)


countryName=temp3['Country Name'].values.tolist()
temp3.drop(['Country Name'], axis=1, inplace=True)
col = temp3.columns
X = temp3.to_numpy()
print(X)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
label = kmeans.labels_

df = pd.DataFrame(X, columns = col)
df['label'] = label
df['country Name'] = countryName


df.to_csv(r"C:\Users\pc\Desktop\countries\countriesClustring\result.csv")
