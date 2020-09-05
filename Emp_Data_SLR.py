##############################################################################
###################### Simple Linear Regression ##############################
##############################################################################






#3) Emp_data -> Build a prediction model for Churn_out_rate 

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library

Emp_Data=pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Simple_linear_Regression\\emp_data.csv")
Emp_Data.columns




######### EDA(explotary data analysis) #################

######### 1st moment business decision ###########
Emp_Data.mean()#Salary_hike       1688.6
                            #Churn_out_rate      72.9
Emp_Data.median()#Salary_hike       1675.0
                  #Churn_out_rate      71.0
Emp_Data.mode()



##########2nd moment busines decision ##############

Emp_Data.var()  #Salary_hike       8481.822222
                 #Churn_out_rate     105.211111                  
Emp_Data.std() #Salary_hike       92.096809
                #Churn_out_rate    10.257247                 

max(Emp_Data['Churn_out_rate'])#92
max(Emp_Data['Salary_hike'])#1870



########### 3rd and 4th moment business decision #########

Emp_Data.skew()#Salary_hike       0.858375
                    #Churn_out_rate    0.647237
Emp_Data.kurt()#Salary_hike       0.165793
               #Churn_out_rate   -0.328199# since the kurtosis value is negative
                                              #implies both the distribution have wider peaks

#### Graphical representation   #########
                  
plt.hist(Emp_Data.Salary_hike)
plt.boxplot(Emp_Data.Salary_hike,0,"rs",0)


plt.hist(Emp_Data.Churn_out_rate)
plt.boxplot(Emp_Data.Churn_out_rate)

plt.plot(Emp_Data.Salary_hike,Emp_Data.Churn_out_rate,"bo");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")


Emp_Data.Churn_out_rate.corr(Emp_Data.Salary_hike) # -0.9117216186909111 # correlation value between X and Y

### or ### table format
Emp_Data.corr()           #               Salary_hike  Churn_out_rate
                         #Salary_hike        1.000000       -0.911722
                          #Churn_out_rate    -0.911722        1.000000

#or using numpy
np.corrcoef(Emp_Data.Salary_hike,Emp_Data.Churn_out_rate)
#array([[ 1.        , -0.91172162],
#       [-0.91172162,  1.        ]])
import seaborn as sns
sns.pairplot(Emp_Data)



########## Renaming the Emp_Data Columns

Emp_Data = Emp_Data.rename(columns={'Salary_hike': 'Salary'}, index={'ONE': 'one'})
Emp_Data = Emp_Data.rename(columns={'Churn_out_rate': 'Churn'}, index={'ONE': 'one'})

############## Model Preparing/ injecting the model #################



# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Churn~Salary",data=Emp_Data).fit()

# For getting coefficients of the varibles used in equation
model.params
#Intercept    244.364911
#Salary        -0.101543
# P-values for the variables and R-squared value for prepared model
model.summary()#Adj. R-squared:                  0.810

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Emp_Data.iloc[:,0]) # Predicted values of Salary using the model
 



# Visualization of regresion line over the scatter plot of Salary and Churn
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Emp_Data['Salary'],y=Emp_Data['Churn'],color='red');plt.plot(Emp_Data['Salary'],pred,color='black');plt.xlabel('Salary');plt.ylabel('Churn')

pred.corr(Emp_Data.Churn) #  0.9117216186909114
# Predicted vs actual values
plt.scatter(x=pred,y=Emp_Data.Churn);plt.xlabel("Predicted");plt.ylabel("Actual")




# Transforming variables for accuracy
model2 = smf.ols('Churn~np.log(Salary)',data=Emp_Data).fit()
model2.params#Intercept         1381.456193
              #np.log(Salary)    -176.109735
model2.summary()#Adj. R-squared:                 0.830

print(model2.conf_int(0.01)) # 99% confidence level

pred2 = model2.predict(pd.DataFrame(Emp_Data['Salary']))

pred2.corr(Emp_Data.Churn)#0.9212077312118845
pred21 = model2.predict(Emp_Data.iloc[:,0])
pred21
plt.scatter(x=Emp_Data['Salary'],y=Emp_Data['Churn'],color='green');plt.plot(Emp_Data['Salary'],pred21,color='blue');plt.xlabel('Salary');plt.ylabel('Churn')




# Exponential transformation
model3 = smf.ols('np.log(Churn)~Salary',data=Emp_Data).fit()
model3.params
model3.summary()# Adj. R-squared:                  0.858
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(Emp_Data['Salary']))
pred_log
pred3=np.exp(pred_log)  
pred3
pred3.corr(Emp_Data.Churn)# 0.9334219364827092
plt.scatter(x=Emp_Data['Salary'],y=Emp_Data['Churn'],color='green');plt.plot(Emp_Data.Salary,np.exp(pred_log),color='blue');plt.xlabel('Salary');plt.ylabel('Churn')
resid_3 = pred3-Emp_Data.Churn#error




# so we will consider the model having highest R-Squared value which is the 1st  model
# getting residuals of the entire data set
Emp_Data_resid = model.resid_pearson #error
Emp_Data_resid 
plt.plot(model.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred,y=Emp_Data.Churn);plt.xlabel("Predicted");plt.ylabel("Actual")


#we can also check the other transformation 
# Quadratic model

model_quad = smf.ols("Churn~Salary+Salary*Salary",data=Emp_Data).fit()
model_quad.params#Intercept    244.364911
                 #Salary        -0.101543
model_quad.summary()#Adj. R-squared:                  0.810
pred_quad = model_quad.predict(Emp_Data.Salary)

model_quad.conf_int(0.05) # 
plt.scatter(Emp_Data.Salary,Emp_Data.Churn,c="b");plt.plot(Emp_Data.Salary,pred_quad,"r")

plt.scatter(np.arange(10),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 


## so we can c without any transformation, the 1st model is giving accuracy having the highest R-squared value












