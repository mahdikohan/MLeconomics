import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt


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

df_gdp = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv')
df_gdp.drop(columns=df_gdp.columns[-1], axis=1, inplace=True)

df_gdppc = pd.read_csv('data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv')
df_gdppc.drop(columns=df_gdppc.columns[-1], axis=1, inplace=True)

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

col_rem = df_oilp.columns
# print(df_oilp[df_oilp['Country Name'].str.contains('Iran')])
df_oilp['avg oil gdp'] = df_oilp[years].sum(axis=1)/len(years)
df_oilp.drop(labels=col_rem, axis=1, inplace=True)
# print(df_oilp[df_oilp['Country Name'].str.contains('Iran')])

# --------------------------------
# Analysis GDPs

temp1 = pd.concat([df_gdp, df_oilp], axis=1)
temp1 = temp1.dropna(subset=all_years, how="any").reset_index(drop=True)

temp2 = temp1.drop(labels=['avg oil gdp'], axis=1)
temp3 = temp1.drop(labels=temp2.columns, axis=1)

mt_gdp_log = np.log(temp2[all_years].to_numpy())

# fill miss values of GDP time series in each country with KNN algorithm
imputer = KNNImputer(n_neighbors=10, weights="uniform")
X = mt_gdp_log
mt_gdp_log = imputer.fit_transform(X)


temp2['log_gdp'] = 0.0
temp2['inter'] = 0.0
temp2['beta'] = 0.0
temp2['r2'] = 0.0

for j in range(len(mt_gdp_log)):
    reg = get_reg(mt_gdp_log[j])
    temp2.at[j, 'gdp'] = mt_gdp_log[j][-1]
    temp2.at[j, 'inter'] = reg[0]
    temp2.at[j, 'beta'] = reg[1]
    temp2.at[j, 'r2'] = reg[2]

temp2.drop(labels=all_years, axis=1, inplace=True)
temp4 = pd.concat([temp2,temp3], axis=1)

print(temp4.columns)

# plt.scatter(temp4['gdp'], temp4['beta'])
# plt.scatter(temp4['inter'], temp4['r2'])
# plt.scatter(temp4['beta'], temp4['avg oil gdp'])
# # plt.scatter(beta, r2)
# # plt.scatter(beta, inter)

plt.show()
