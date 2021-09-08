# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
# from pandas_profiling import ProfileReport
# import numpy as np
# from sklearn.preprocessing import PowerTransformer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.linear_model import LinearRegression
#from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
# import requests
# from bs4 import BeautifulSoup as bs
# from urllib.request import urlopen as uReq
# from sklearn.model_selection import train_test_split
import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso, RidgeCV,LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
import lg
lggg = lg.logg()

ln=0.0
elst=0.0

from flask import Flask, request, url_for, redirect, render_template

app = Flask(__name__)






app = Flask(__name__)  # initialising the flask app with the name 'app'

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    ln = ln_reg()
    ln.dropfeatures()
    ln.modeling()
    data = {"ln":100,"ridge":200}
    return render_template("index.html", content=data)


@app.route('/initaldata', methods=['GET', 'POST'])
def ini():
    data = 1212
    return render_template('initaldata.html', content=data)

@app.route('/After EDA', methods=['GET', 'POST'])
def Final():
    return render_template('After EDA.html')


class ln_reg:

    def __init__(self):
        try:
            lggg.info('Reading CSV file')
            self.df = pd.read_csv("ai4i2020.csv")
            # self.initialreport()
        except Exception as e:
            lggg.excpt(f'exception had occured during file read operation. i.e., {e}')

    # def initialreport(self):
    #     try:
    #         lggg.info('Initial report is preparing')
    #         self.pf = ProfileReport(self.df)
    #         self.pf.to_file('initaldata.html')
    #         self.dropfeatures()
    #     except Exception as e:
    #         lggg.excpt(f'Exception {e} had occured during report generation')

    """# Following are the Data analysis 
    1. As per data set. air temperature (y) is highly correlated with process temperature. model will take advantage of it.
    2. As rotational speed is highly correlated with torque. we can drop rotational speed. Additionally rotational speed was a right skwed data set
    3. random failures (RNF) is not related with target y value. therefore, we will drop it.
    4. Machine failure is related with many feature. So we will drop it due to multi-collinearity
    5. As per Phik correlation coefficient torque and PWF are highly related. So we drop PWF
    6. Try to generate gussian like data for tool wear using sklearn.preprocessing.PowerTransformer
    7. Torque, Tool wear, TWF, HDF & OSF have null or negligible relation with dependent variable"""

    def dropfeatures(self):
        try:
            lggg.info('Droping unnecceasry data')
            self.df1 = self.df.drop(axis=1,columns=['Rotational speed [rpm]','Machine failure','RNF',
                                          'UDI','Product ID','Type','PWF','Torque [Nm]','Tool wear [min]','TWF','HDF','OSF'])
            # self.finalreport()
        except Exception as e:
            lggg.excpt(f'Exception {e} had occured during droping features')

    # def finalreport(self):
    #     try:
    #         lggg.info("Final report is preparing")
    #         ProfileReport(self.df1).to_file('After EDA.html')
    #         self.modeling()
    #     except Exception as e:
    #         lggg.excpt(f'Exception {e} had occured during final report preparation')

    def modeling(self):
        try:
            global ln
            global elst
            self.x = self.df1[['Process temperature [K]']]
            self.y = self.df1[['Air temperature [K]']]
            """ from linear regression"""
            self.linear = LinearRegression()

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25)
            self.linear.fit(self.x_train,self.y_train)
            self.file = 'linear_reg.sav'
            # pickle.dump(self.linear,open(self.file,'wb'))
            ln = self.linear.score(self.x_test,self.y_test)
            lggg.info(f"Score of linear regression model is {ln}")
            ##from ElasticNet
            self.elastic = ElasticNetCV(alphas=None,cv=10)
            self.elastic.fit(self.x_train,self.y_train)
            self.elastic_lr = ElasticNet(alpha=self.elastic.alpha_, l1_ratio=self.elastic.l1_ratio_)
            self.elastic_lr.fit(self.x_train,self.y_train)

            elst= self.elastic_lr.score(self.x_test,self.y_test)
            lggg.info(f"Score of ElasticNet model is {elst}")

        except  Exception as e:
            lggg.excpt(f"Exception (i.e.,{e} is occur during modeling")
@app.route('/Score', methods=['GET', 'POST'])
def score():
    data= {"linear regression Score is": ln,
           "ElasticNet regression Score is": elst}
    return render_template('cool_form.html', content=data)

#ln = ln_reg("ai4i2020.csv")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()

