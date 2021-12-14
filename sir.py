import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

date=["2/20/2021","2/21/2021","2/22/2021","2/23/2021","2/24/2021","2/25/2021","2/26/2021",
      "2/27/2021","2/28/2021","3/1/2021","3/2/2021","3/3/2021","3/4/2021","3/5/2021"]
new=[8054,7300,10180,9775,7533,8493,8232,6208,5560,6680,5712,6608,7264,6971]
heal=[9835,8236,9918,7996,7735,8686,7261,7382,6649,9212,8948,9053,6440,6331]
die=[164,173,202,323,240,264,268,195,185,159,193,203,176,129]
t_case=[8054,15354,25534,35309,42842,51335,59567,65775,71335,78015,83727,90535,97799,104770]
def data_spilt(data, orders, start):
    x_train = np.empty((len(data) - start - orders, orders))
    y_train = data[start + orders:]

    for i in range(len(data) - start - orders):
        x_train[i] = data[i + start:start + orders + i]

    return x_train, y_train

########## data ##########
# X_cml = cumulative confirmed cases
X_cml = np.array([8054,7300,10180,9775,7533,8493,8232,6208,5560,6680,5712,6608,7264,6971], dtype=np.float64)
# recovered = cumulative recovered cases
recovered = np.array([9835,8236,9918,7996,7735,8686,7261,7382,6649,9212,8948,9053,6440,6331], dtype=np.float64)
# death = cumulative deaths
death = np.array([164,173,202,323,240,264,268,195,185,159,193,203,176,129], dtype=np.float64)

population = 276361788
########## data preprocess ##########
X = X_cml - recovered - death
R = recovered + death

n = np.array([population] * len(X), dtype=np.float64)

S = n - X - R

X_diff = np.array([X[:-1], X[1:]], dtype=np.float64).T
R_diff = np.array([R[:-1], R[1:]], dtype=np.float64).T

gamma = (R[1:] - R[:-1]) / X[:-1]
beta = n[:-1] * (X[1:] - X[:-1] + R[1:] - R[:-1]) / (X[:-1] * (n[:-1] - X[:-1] - R[:-1]))
R0 = beta / gamma

########## Parameters for Ridge Regression ##########
orders_beta = 3
orders_gamma = 3

##### Select a starting day for the data training in the ridge regression. #####
start_beta = 0
start_gamma = 0


print("The latest basic reproduction number R0:", R0[-1])

########## Ridge Regression ##########
##### Split the data to the training set and testing set #####
x_beta, y_beta = data_spilt(beta, orders_beta, start_beta)
x_gamma, y_gamma = data_spilt(gamma, orders_gamma, start_gamma)

clf_beta = Ridge(alpha=0.003765, copy_X=True, fit_intercept=False, max_iter=None, normalize=True, random_state=None, 
solver='auto', tol=1e-08).fit(x_beta, y_beta)
clf_gamma = Ridge(alpha=0.001675, copy_X=True, fit_intercept=False, max_iter=None,normalize=True, random_state=None, 
solver='auto', tol=1e-08).fit(x_gamma, y_gamma)

beta_hat = clf_beta.predict(x_beta)
gamma_hat = clf_gamma.predict(x_gamma)
stop_X = 0 # stopping criteria
stop_day = 100 # maximum iteration days 

day_count = 0
turning_point = 0

S_predict = [S[-1]]
X_predict = [X[-1]]
R_predict = [R[-1]]

predict_beta = np.array(beta[-orders_beta:]).tolist()
predict_gamma = np.array(gamma[-orders_gamma:]).tolist()
while (X_predict[-1] >= stop_X) and (day_count <= stop_day):
    if predict_beta[-1] > predict_gamma[-1]:
        turning_point += 1

    next_beta = clf_beta.predict(np.asarray([predict_beta[-orders_beta:]]))[0]
    next_gamma = clf_gamma.predict(np.asarray([predict_gamma[-orders_gamma:]]))[0]

    if next_beta < 0:
        next_beta = 0
    if next_gamma < 0:
        next_gamma = 0

    predict_beta.append(next_beta)
    predict_gamma.append(next_gamma)

    next_S = ((-predict_beta[-1] * S_predict[-1] *
               X_predict[-1]) / n[-1]) + S_predict[-1]
    next_X = ((predict_beta[-1] * S_predict[-1] * X_predict[-1]) /
              n[-1]) - (predict_gamma[-1] * X_predict[-1]) + X_predict[-1]
    next_R = (predict_gamma[-1] * X_predict[-1]) + R_predict[-1]

    S_predict.append(next_S)
    X_predict.append(next_X)
    R_predict.append(next_R)

    day_count += 1

########## Print Info ##########
print('\nConfirmed cases tomorrow:', np.rint(X_predict[1] + R_predict[1]))
print('Infected persons tomorrow:', np.rint(X_predict[1]))
print('Recovered + Death persons tomorrow:', np.rint(R_predict[1]))
print('\nEnd day:', day_count)
print('Confirmed cases on the end day:', np.rint(X_predict[-2] + R_predict[-2]))

#print('\nTuring point:', turning_point)

def case(date,case):
    plt.plot(date,new,label=("Case(New)"))
    plt.legend()
def healed(date,heal):
    plt.plot(date,heal,label=("Healed"))
    plt.legend()
def died(date,die):
    plt.plot(date,die,label=("Die"))
    plt.legend()
def total(date,t_case):
    plt.plot(date,t_case,label=("Total case"))
    plt.legend()
plt.title("Graphic")
plt.xlabel("Date")
plt.ylabel("People")
case(date,new)
healed (date,heal)
died(date,die)
total(date,t_case)
plt.show()