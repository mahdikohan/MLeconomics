import pandas as pd
import numpy as np
from sklearn import linear_model
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

df_import = pd.read_csv('./API_BM.GSR.GNFS.CD_DS2_en_csv_v2_6304000.csv')
df_import.drop(columns=df_import.columns[-1], axis=1, inplace=True)

df_export = pd.read_csv('./API_NE.EXP.GNFS.CD_DS2_en_csv_v2_6300789.csv')
df_export.drop(columns=df_export.columns[-1], axis=1, inplace=True)

df_gdp = pd.read_csv('./API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv')
df_gdp.drop(columns=df_gdp.columns[-1], axis=1, inplace=True)

df_gdppc = pd.read_csv('./API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv')
df_gdppc.drop(columns=df_gdppc.columns[-1], axis=1, inplace=True)

df_oilp = pd.read_csv('./API_NY.GDP.PETR.RT.ZS_DS2_en_csv_v2_6305850.csv')
df_oilp.drop(columns=df_oilp.columns[-1], axis=1, inplace=True)

df_gini = pd.read_csv('./API_SI.POV.GINI_DS2_en_csv_v2_6508497.csv')
df_gini.drop(columns=df_gini.columns[-1], axis=1, inplace=True)

df_deathr = pd.read_csv('./API_SP.DYN.CDRT.IN_DS2_en_csv_v2_6303594.csv')
df_deathr.drop(columns=df_deathr.columns[-1], axis=1, inplace=True)

# --------------------------------


# --------------------------------
# Analysis oil

# df_oilp_data = df_oilp.iloc[:,15:-1]
df_oilp_data = df_oilp.iloc[:,-10:-1].sum(axis=1)/10
df_oilp_data = df_oilp_data.reset_index()
df_oilp_data.columns = ["index","avg oil"]
df_oilp_data.drop(columns=["index"], inplace=True)

df_oilp_data = pd.concat([df_gdp, df_oilp_data], axis=1)

print(df_oilp_data)
# country_list = df_oilp.iloc[:,:4]

# df_oilp_data = pd.concat([country_list, df_oilp_data], axis=1)

# # print(type(df_oilp_data))
# print(df_oilp_data)
# # plt.plot(df_oilp_data)
# # plt.show()

# --------------------------------
# Analysis export


# --------------------------------
# Analysis GDP per capita
# print(df_export)

# --------------------------------
# Analysis GDPs

# temp1 = df_gdp.dropna(how="any").reset_index(drop=True)

# df_gdp_log = temp1.iloc[:,4:]
# country_list = temp1.iloc[:,:4]
# gdp_log = np.log(df_gdp_log.to_numpy())

# gdp = []
# inter = []
# beta = []
# r2 = []

# for j in range(len(gdp_log)):
#     country = country_list.iloc[j,0]
#     reg = get_reg(gdp_log[j])

#     gdp.append(gdp_log[j][-1])
#     inter.append(reg[0])
#     beta.append(reg[1])
#     r2.append(reg[2])

#     print(country,reg)

# plt.scatter(gdp, beta)
# # plt.scatter(inter, r2)
# # plt.scatter(beta, r2)
# # plt.scatter(beta, inter)

# plt.show()
