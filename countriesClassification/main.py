import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def get_reg(X,y):
    
# Load and transform data --------

df_import = pd.read_csv('./API_BM.GSR.GNFS.CD_DS2_en_csv_v2_6304000.csv')
df_import.drop(columns=df_import.columns[-1], axis=1, inplace=True)

df_export = pd.read_csv('./API_NE.EXP.GNFS.CD_DS2_en_csv_v2_6300789.csv')
df_export.drop(columns=df_export.columns[-1], axis=1, inplace=True)

df_gdp = pd.read_csv('./API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv')
df_gdp.drop(columns=df_gdp.columns[-1], axis=1, inplace=True)

df_gdppc = pd.read_csv('./API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv')
df_gdppc.drop(columns=df_gdppc.columns[-1], axis=1, inplace=True)

df_gini = pd.read_csv('./API_SI.POV.GINI_DS2_en_csv_v2_6508497.csv')
df_gini.drop(columns=df_gini.columns[-1], axis=1, inplace=True)

df_deathr = pd.read_csv('./API_SP.DYN.CDRT.IN_DS2_en_csv_v2_6303594.csv')
df_deathr.drop(columns=df_deathr.columns[-1], axis=1, inplace=True)


# --------------------------------

temp1 = df_gdp.dropna(how="any").reset_index(drop=True)
# temp1.to_csv(r"C:\Users\pc\Desktop\countries classifiers\gdp.csv")
df_gdp_log = temp1.iloc[:,4:]
country_list = temp1.iloc[:,:4]

# print(country_list)

# --------------------------------

gdp_log = np.log(df_gdp_log.to_numpy())
reg = linear_model.LinearRegression()

X = np.arange(len(gdp_log[6])).reshape(-1,1)
y = gdp_log[6].reshape(-1,1)
reg.fit(X, y)

y_pred = reg.predict(X)

print(f"w0:{reg.intercept_[0]}, w1:{reg.coef_[0][0]}")

# Plot

plt.plot(gdp_log[6])
plt.plot(X, y_pred, color='red')
plt.show()


# plt.legend(country_list["Country Name"][6:9].values.tolist())


plt.show()
# print(gdp_log[0])



# print(len(df_gdp_log.values))
# print(len(df_deathr))
# print(df_imports.columns[-1])
