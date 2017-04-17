import numpy as np
import statsmodels.api as sm
import cvxpy as cvx

def predict_liabilities(liability_history, T,method="MA2"):
    if method == "MA2":
        model = sm.tsa.ARMA(liability_history, (0,2))
    elif method == "AR1":
        model = sm.tsa.ARMA(liability_history, (1, 0))
    else:
        raise IOError, "Method not supported, please input method='AR1' or method='MA2'"
    result = model.fit()
    estimated_params = result.params
    forecast = result.forecast(T)
    return estimated_params,forecast[0]

def naive_strategy(c, A, b):
    v = cvx.Variable(len(c))
    gain = c.T*v
    const = []
    for i in range(len(c)):
        const += [v[i] >= 0]
        if i < 5:
            const += [v[i] <= 1]
    for i in range(len(b)):
        const += [A[i, :].T * v == b[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

###def robust_strategy(c,A,b_hat,params):

###def affine_recourse(c,A,b_hat,params):

def evaluate_strategy(best_case_obj, naive_strategy_obj, robust_strategy_obj, affine_recourse_obj):
    if naive_strategy_obj is not None:
        regret_naive_strategy = np.square(best_case_obj - naive_strategy_obj)
    else:
        regret_naive_strategy = 1.0E6
    regret_robust_strategy = best_case_obj - robust_strategy_obj
    regret_affine_recourse = best_case_obj - affine_recourse_obj
    return np.argmin(regret_naive_strategy, regret_robust_strategy, regret_affine_recourse)

def run(liability_history, T, c, A, b_real, predict_method = "MA2"):
    params, liability_forecast = predict_liabilities(liability_history, T, predict_method)
    best_strategy_return, best_strategy_opt_values = naive_strategy(c, A, b_real)
    naive_strategy_return, naive_strategy_opt_values = naive_strategy(c, A, liability_forecast)
    '''robust_strategy_return, robust_strategy_opt_values = robust_strategy(c, A, ma_2_liability_forecast, params)
    affine_recourse_return, affine_recourse_opt_values = affine_recourse(c, A, ma_2_liability_forecast, params)
    best_strategy_index = evaluate_strategy(best_strategy_return, naive_strategy_return,
                                            robust_strategy_return, affine_recourse_return)'''

if __name__ == '__main__':
    all_liability_history = np.loadtxt('projectdata.txt')
    training_data = all_liability_history[:-6]
    T = 6
    c = np.append(np.zeros(13),1)
    A = np.array([[1,0,0,0,0,1,0,0,-1,0,0,0,0,0],
                  [-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0,0],
                  [0,-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0],
                  [0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0,0],
                  [0,0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0],
                  [0,0,0,0,-1.01,0,0,-1.02,0,0,0,0,1.003,-1]])
    b_real = all_liability_history[-6:]
    predict_method = 'MA2'

    run(training_data, T, c, A, b_real, predict_method)