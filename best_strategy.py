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
    B = np.diag(b_hat).dot(np.array([[alpha[0], 0, 0, 0, 0, 0],
                                     [alpha[1], alpha[0], 0, 0, 0, 0],
                                     [alpha[2], alpha[1], alpha[0], 0, 0, 0],
                                     [alpha[3], alpha[2], alpha[1], alpha[0], 0, 0],
                                     [alpha[4], alpha[3], alpha[2], alpha[1], alpha[0], 0],
                                     [alpha[5], alpha[4], alpha[3], alpha[2], alpha[1], alpha[0]]]))
    return B

def robust_strategy(c,A,b_hat,params, initial_cash, method = "MA2"):
    B = get_B(b_hat, params, method)
    B_norm = np.linalg.norm(B,ord=1,axis=1)
    v = cvx.Variable(len(c))
    gain = c.T * v
    const = []
    for i in range(len(c)):
        const += [v[i] >= 0]
        if i < 5:
            const += [v[i] <= 1]
    const += [A[0, :].T * v == b_hat[0] + B_norm[0] - initial_cash]
    for i in range(1, len(b_hat)):
        const += [A[i, :].T * v >= b_hat[i] + B_norm[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def affine_recourse(c,A,b_hat,params, initial_cash, method = "MA2"):
    B = get_B(b_hat, params, method)
    v = cvx.Variable(len(c))
    V = cvx.Variable(rows=len(c), cols=len(A))
    gain = c.T * v - cvx.norm1(V.T*c)
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
    const += [A[0, :].T * v >= b_hat[0] + B_norm - initial_cash]
    for i in range(1, len(b_hat)):
        B_norm = cvx.norm1(B_minus_AV[i,:])
        const += [A[i, :].T * v >= b_hat[i] + B_norm]
    for i in range(len(c)):
        V_norm = cvx.norm1(V[i,:])
        const += [v[i] >= V_norm]
        if i < 5:
            const += [v[i] + V_norm <= 1]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def check_feasible(A,v,b, initial_cash):
    Av = A.dot(v)
    if not np.isclose(Av[0], b[0] - initial_cash) and Av[0] < b[0] - initial_cash:
        return False
    for i in range(1, len(b)):
        if not np.isclose(Av[i], b[i]) and Av[i] < b[i]:
            return False
    return True

def evaluate_strategy(best_case_obj, naive_strategy_obj, robust_strategy_obj, affine_recourse_obj,
                      is_feasible_naive, is_feasible_robust, is_feasible_affine_recourse):
    if not is_feasible_naive:
        regret_naive_strategy = 1.0E6
    else:
        regret_naive_strategy = best_case_obj - naive_strategy_obj
    if not is_feasible_robust:
        regret_robust_strategy = 1.0E6
    else:
        regret_robust_strategy = best_case_obj - robust_strategy_obj
    if not is_feasible_affine_recourse:
        regret_affine_recourse = 1.0E6
    else:
        regret_affine_recourse = best_case_obj - affine_recourse_obj
    if regret_naive_strategy == 1.0E6 and regret_robust_strategy == 1.0E6 and regret_affine_recourse == 1.0E6:
        return 3
    return np.argmin(np.array([regret_naive_strategy, regret_robust_strategy, regret_affine_recourse]))

def run(liability_history, initial_cash, T, c, A, b_real, predict_method = "MA2"):
    params, liability_forecast = predict_liabilities(liability_history, T, predict_method)
    best_strategy_return, best_strategy_opt_values = naive_strategy(c, A, b_real, initial_cash)
    naive_strategy_return, naive_strategy_opt_values = naive_strategy(c, A, liability_forecast, initial_cash)
    robust_strategy_return, robust_strategy_opt_values = robust_strategy(c, A, liability_forecast, params, initial_cash)
    affine_recourse_return, affine_recourse_opt_values = affine_recourse(c, A, liability_forecast, params, initial_cash)
    is_feasible_naive = check_feasible(A, naive_strategy_opt_values, b_real, initial_cash)
    is_feasible_robust = check_feasible(A, robust_strategy_opt_values, b_real,initial_cash)
    is_feasible_affine_recourse = check_feasible(A,affine_recourse_opt_values, b_real, initial_cash)
    best_strategy_index = evaluate_strategy(best_strategy_return, naive_strategy_return,
                                            robust_strategy_return, affine_recourse_return,
                      is_feasible_naive, is_feasible_robust, is_feasible_affine_recourse)
    dict = {0: "Naive strategy", 1: "Robust strategy", 2: "Affine Recourse strategy"}
    if best_strategy_index < 3:
        print "Best strategy is "+dict.get(best_strategy_index)
    else:
        print "None of the strategies would have been feasible"

if __name__ == '__main__':
    all_liability_history = np.loadtxt('projectdata.txt')
    initial_cash = 70.3
    T = 6
    c = np.append(np.zeros(13),1)
    A = np.array([[1,0,0,0,0,1,0,0,-1,0,0,0,0,0],
                  [-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0,0],
                  [0,-1.01,1,0,0,0,0,1,0,1.003,-1,0,0,0],
                  [0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0,0],
                  [0,0,0,-1.01,1,0,-1.02,0,0,0,0,1.003,-1,0],
                  [0,0,0,0,-1.01,0,0,-1.02,0,0,0,0,1.003,-1]])
    predict_method = 'MA2'

    for i in range(54,6,-1):
        try:
            training_data = all_liability_history[:-i]
            b_real = all_liability_history[-i:-i+6]
            run(training_data, initial_cash, T, c, A, b_real, predict_method)
        except:
            print "Infeasible solution in one of the strategies"