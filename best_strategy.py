import numpy as np
import statsmodels.api as sm
import cvxpy as cvx

def predict_liabilities(liability_history, T,method="MA2"):
    if method == "MA2":
        model = sm.tsa.ARMA(liability_history, (0,2))
    elif method == "AR1":
        model = sm.tsa.ARMA(liability_history, (1, 0))
    else:
        raise IOError, "Method not supported, please input method='AR1'" \
                       " or method='MA2'"
    result = model.fit()
    estimated_params = result.params
    forecast = result.forecast(T)
    return estimated_params,forecast[0]

def naive_strategy(c, A, b, initial_cash):
    v = cvx.Variable(len(c))
    gain = c.T*v
    const = []
    for i in range(len(c)):
        const += [v[i] >= 0]
        if i < 5:
            const += [v[i] <= 1]
    const += [A[0, :].T * v == b[0] - initial_cash]
    for i in range(1,len(b)):
        const += [A[i, :].T * v == b[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def get_B(b_hat, params, method = "MA2"):
    mu = params[0]
    alpha = np.zeros(len(A))
    if method == "AR1":
        theta_1 = params[1]
        alpha[0] = 1.0 / mu * (1 + theta_1)
        alpha[1] = theta_1 * alpha[0]
    if method == "MA2":
        alpha[0] = 1.0 / mu
        alpha[1] = params[1] / mu
        alpha[2] = params[2] / mu
    B = np.diag(b_hat).dot(np.array
                           ([[alpha[0], 0, 0, 0, 0, 0],
                            [alpha[1], alpha[0], 0, 0, 0, 0],
                            [alpha[2], alpha[1], alpha[0], 0, 0, 0],
                            [alpha[3], alpha[2], alpha[1], alpha[0], 0, 0],
                            [alpha[4], alpha[3], alpha[2], alpha[1], alpha[0], 0],
                            [alpha[5], alpha[4], alpha[3], alpha[2], alpha[1], alpha[0]]]))
    return B

def robust_strategy(c,A,b_hat,params, initial_cash, sigma, method = "MA2"):
    B = get_B(b_hat, params, method)
    B_norm = np.linalg.norm(B,ord=1,axis=1)
    v = cvx.Variable(len(c))
    gain = c.T * v
    const = []
    for i in range(len(c)):
        const += [v[i] >= 0]
        if i < 5:
            const += [v[i] <= 1]
    const += [A[0, :].T * v == b_hat[0] + sigma*B_norm[0] - initial_cash]
    for i in range(1, len(b_hat)):
        const += [A[i, :].T * v >= b_hat[i] + sigma*B_norm[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def affine_recourse(c,A,b_hat,params, initial_cash, sigma, method = "MA2"):
    B = get_B(b_hat, params, method)
    v = cvx.Variable(len(c))
    V = cvx.Variable(rows=len(c), cols=len(A))
    gain = c.T * v - sigma*cvx.norm1(V.T*c)
    const = []
    for i in range(5):
        for j in range(len(A)):
            if i <= j:
                const += [V[i,j] == 0]
    for i in range(3):
        for j in range(len(A)):
            if i <= j:
                const += [V[i+5,j] == 0]
    for i in range(6):
        for j in range(len(A)):
            if i <= j:
                const += [V[i+8,j] == 0]
    B_minus_AV = B - A*V
    B_norm = cvx.norm1(B_minus_AV[0,:])
    const += [A[0, :].T * v >= b_hat[0] + sigma*B_norm - initial_cash]
    for i in range(1, len(b_hat)):
        B_norm = cvx.norm1(B_minus_AV[i,:])
        const += [A[i, :].T * v >= b_hat[i] + sigma*B_norm]
    for i in range(len(c)):
        V_norm = cvx.norm1(V[i,:])
        const += [v[i] >= sigma*V_norm]
        if i < 5:
            const += [v[i] + sigma*V_norm <= 1]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def check_feasible(A,v,b, initial_cash):
    Av = A.dot(v)
    if not np.isclose(Av[0], b[0] - initial_cash) and Av[0] < b[0] - \
            initial_cash:
        return False
    for i in range(1, len(b)):
        if not np.isclose(Av[i], b[i]) and Av[i] < b[i]:
            return False
    return True

def evaluate_strategy(best_case_obj, naive_strategy_obj,
                      robust_strategy_obj, affine_recourse_obj,
                      is_feasible_naive, is_feasible_robust,
                      is_feasible_affine_recourse):
    if not is_feasible_naive:
        regret_naive_strategy = best_case_obj
    else:
        regret_naive_strategy = best_case_obj - naive_strategy_obj
    if not is_feasible_robust:
        regret_robust_strategy = best_case_obj
    else:
        regret_robust_strategy = best_case_obj - robust_strategy_obj
    if not is_feasible_affine_recourse:
        regret_affine_recourse = best_case_obj
    else:
        regret_affine_recourse = best_case_obj - affine_recourse_obj
    if regret_naive_strategy == regret_robust_strategy \
            and  regret_naive_strategy == regret_affine_recourse:
        best_strategy_index = 3
    else:
        best_strategy_index = np.argmin(np.array([
            regret_naive_strategy,
            regret_robust_strategy,
            regret_affine_recourse]))
    return regret_naive_strategy, regret_robust_strategy, \
           regret_affine_recourse, best_strategy_index

def run(liability_history, initial_cash, T, c, A, b_real,
        sigma_robust, sigma_affine_recourse, predict_method = "MA2"):
    params, liability_forecast = predict_liabilities(
        liability_history, T, predict_method)
    best_strategy_return, best_strategy_opt_values = \
        naive_strategy(c, A, b_real, initial_cash)
    naive_strategy_return, naive_strategy_opt_values = \
        naive_strategy(c, A, liability_forecast, initial_cash)
    robust_strategy_return, robust_strategy_opt_values = \
        robust_strategy(c, A, liability_forecast, params, initial_cash,
                        sigma_robust, predict_method)
    affine_recourse_return, affine_recourse_opt_values = \
        affine_recourse(c, A, liability_forecast, params,
                        initial_cash, sigma_affine_recourse, predict_method)
    if not naive_strategy_return:
        is_feasible_naive = False
    else:
        is_feasible_naive = check_feasible(A, naive_strategy_opt_values,
                                       b_real, initial_cash)
    if not robust_strategy_return:
        is_feasible_robust = False
    else:
        is_feasible_robust = check_feasible(A, robust_strategy_opt_values,
                                        b_real,initial_cash)
    if not affine_recourse_return:
        is_feasible_affine_recourse = False
    else:
        is_feasible_affine_recourse = check_feasible(A,
                            affine_recourse_opt_values, b_real, initial_cash)
    return evaluate_strategy(best_strategy_return, naive_strategy_return,
                        robust_strategy_return, affine_recourse_return,
                      is_feasible_naive, is_feasible_robust, is_feasible_affine_recourse)

if __name__ == '__main__':
    all_liability_history = np.loadtxt('projectdata.txt')
    initial_cash = 70.3
    sigma_robust = 3.0
    sigma_affine_recourse = 3.0
    T = 6
    c = np.append(np.zeros(13),1)
    A = np.array([[1,0,0,0,0,1,0,0,-1,0,0,0,0,0],
                  [-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0,0],
                  [0,-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0],
                  [0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0,0],
                  [0,0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0],
                  [0,0,0,0,-1.01,0,0,-1.02,0,0,0,0,1.003,-1]])
    predict_method = 'AR1'

    dict_expected_regret = {"Naive strategy regret":0,
                            "Robust strategy regret":0,
                            "Affine Recourse regret":0}
    dict_best_strategy = {"Naive strategy": 0,
                          "Robust strategy": 0,
                          "Affine Recourse": 0,
                          "None feasible": 0}
    count = 0
    for i in range(54,6,-1):
        count += 1
        try:
            training_data = all_liability_history[:-i]
            b_real = all_liability_history[-i:-i+6]
            regret_naive_strategy, regret_robust_strategy, regret_affine_recourse, \
            best_strategy_index = \
                run(training_data, initial_cash, T, c, A, b_real, sigma_robust,
                    sigma_affine_recourse, predict_method)
            dict_expected_regret["Naive strategy regret"] += regret_naive_strategy
            dict_expected_regret["Robust strategy regret"] += regret_robust_strategy
            dict_expected_regret["Affine Recourse regret"] += regret_affine_recourse
            if best_strategy_index == 0:
                dict_best_strategy["Naive strategy"] += 1
            elif best_strategy_index == 1:
                dict_best_strategy["Robust strategy"] +=1
            elif best_strategy_index == 2:
                dict_best_strategy["Affine Recourse"] += 1
            elif best_strategy_index == 3:
                dict_best_strategy["None feasible"] += 1
        except:
            continue
    dict_expected_regret["Naive strategy regret"] /= count
    dict_expected_regret["Robust strategy regret"] /= count
    dict_expected_regret["Affine Recourse regret"] /= count
    print dict_expected_regret
    print dict_best_strategy