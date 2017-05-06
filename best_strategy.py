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
        if i < len(b) - 1:
            const += [v[i] <= 1]
    const += [A[0, :].T * v == b[0] - initial_cash]
    for i in range(1,len(b)):
        const += [A[i, :].T * v == b[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def get_B(b, params, T, predict_method = 'MA2'):
    mu = params[0]
    alpha = np.zeros(T)
    if predict_method  == "AR1":
        theta_1 = params[1]
        alpha[0] = 1.0 / mu * (1 + theta_1)
        alpha[1] = theta_1 * alpha[0]
    if predict_method == "MA2":
        alpha[0] = 1.0 / mu
        alpha[1] = params[1] / mu
        alpha[2] = params[2] / mu
    alpha_matrix = np.array([[alpha[0], 0, 0, 0, 0, 0],
                            [alpha[1], alpha[0], 0, 0, 0, 0],
                            [alpha[2], alpha[1], alpha[0], 0, 0, 0],
                            [alpha[3], alpha[2], alpha[1], alpha[0], 0, 0],
                            [alpha[4], alpha[3], alpha[2], alpha[1], alpha[0], 0],
                            [alpha[5], alpha[4], alpha[3], alpha[2], alpha[1], alpha[0]]])
    B = np.diag(b).dot(alpha_matrix[:len(b),:len(b)])
    return B

def robust_strategy(c,A,b,initial_cash, B, sigma):
    B_norm = np.linalg.norm(B,ord=1,axis=1)
    v = cvx.Variable(len(c))
    gain = c.T * v
    const = []
    for i in range(len(c)):
        const += [v[i] >= 0]
        if i < len(b) - 1:
            const += [v[i] <= 1]
    const += [A[0, :].T * v == b[0] + sigma*B_norm[0] - initial_cash]
    for i in range(1, len(b)):
        const += [A[i, :].T * v >= b[i] + sigma*B_norm[i]]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def affine_recourse(c,A,b, initial_cash, B, sigma):
    v = cvx.Variable(len(c))
    V = cvx.Variable(rows=len(c), cols=len(A))
    gain = c.T * v - sigma*cvx.norm1(V.T*c)
    const = []
    for i in range(len(b)-1):
        for j in range(len(A)):
            if i <= j:
                const += [V[i,j] == 0]
    if len(b) > 3:
        for i in range(len(b) - 3):
            for j in range(len(A)):
                if i <= j:
                    const += [V[i+len(b)-1,j] == 0]
        for i in range(len(b)):
            for j in range(len(A)):
                if i <= j:
                    const += [V[i+2*len(b) - 4,j] == 0]
    elif len(b) > 1:
        for i in range(len(b)):
            for j in range(len(A)):
                if i <= j:
                    const += [V[i + len(b) - 1, j] == 0]
    B_minus_AV = B - A*V
    B_norm = cvx.norm1(B_minus_AV[0,:])
    const += [A[0, :].T * v >= b[0] + sigma*B_norm - initial_cash]
    for i in range(1, len(b)):
        B_norm = cvx.norm1(B_minus_AV[i,:])
        const += [A[i, :].T * v >= b[i] + sigma*B_norm]
    for i in range(len(c)):
        V_norm = cvx.norm1(V[i,:])
        const += [v[i] >= sigma*V_norm]
        if i < len(b) - 1:
            const += [v[i] + sigma*V_norm <= 1]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def affine_cash(c,A,b, initial_cash, B, sigma):
    v = cvx.Variable(len(c))
    X = cvx.Variable(rows=len(b)-1, cols=len(A))
    gain = c.T * v - sigma * cvx.norm1(X.T * c[:len(b)-1])
    const = []
    for i in range(len(b) - 1):
        for j in range(len(A)):
            if i <= j:
                const += [X[i, j] == 0]
    B_minus_AV = B - A[:,:len(b) - 1] * X
    B_norm = cvx.norm1(B_minus_AV[0, :])
    const += [A[0, :].T * v >= b[0] + sigma * B_norm - initial_cash]
    for i in range(1, len(b)):
        B_norm = cvx.norm1(B_minus_AV[i, :])
        const += [A[i, :].T * v >= b[i] + sigma * B_norm]
    for i in range(len(b) - 1):
        X_norm = cvx.norm1(X[i, :])
        const += [v[i] >= sigma * X_norm]
        const += [v[i] + sigma * X_norm <= 1]
    obj = cvx.Maximize(gain)
    prob = cvx.Problem(objective=obj, constraints=const)
    prob.solve()
    return obj.value, v.value

def is_feasibile(A,opt_values,cash, liability):
    if opt_values is None:
        return False
    return A[0, :].T * opt_values + cash >= liability

def update_cash(A,opt_values,cash,liability,T, t, y_opt):
    if not is_feasibile(A, opt_values, cash, liability):
        cash = -1.0E6
    elif t == T-1:
        cash = opt_values
    else:
        x_opt_cash = 1.01 * opt_values[0]
        if t <= 2:
            z_opt_cash = 1.003 * opt_values[T - 2 * t + 2]
            y_opt[t] = 1.02 * opt_values[T - t - 1]
            cash = A[0, :].T * opt_values + cash - liability - x_opt_cash + z_opt_cash
        else:
            z_opt_cash = 1.003 * opt_values[5 - t]
            cash = A[0, :].T * opt_values + cash - liability - x_opt_cash - \
                   y_opt[t - 3] + z_opt_cash
    return cash, y_opt


def run(c, A, b_real, T, cash_naive, cash_robust, cash_affine_recourse,
        cash_affine_cash,
        all_liability_history, sigma_robust,
        sigma_affine_recourse, predict_method='MA2'):
    best_strategy_return, best_strategy_opt_values = \
        naive_strategy(c, A, b_real, cash_naive)
    regret_naive_strategy, regret_robust_strategy, \
    regret_affine_recourse = 0, 0, 0
    regret_affine_cash = 0
    y_opt_cash_naive = np.zeros(3)
    y_opt_cash_robust = np.zeros(3)
    y_opt_cash_affine_recourse = np.zeros(3)
    y_opt_cash_affine_cash = np.zeros(3)
    for i in range(T):
        params, b = predict_liabilities(np.append(
            all_liability_history,b_real[:i]), T-i, predict_method)
        if regret_naive_strategy < best_strategy_return:
            naive_strategy_return, naive_strategy_opt_values =\
                naive_strategy(c, A, b, cash_naive)
            cash_naive, y_opt_cash_naive = \
                update_cash(A, naive_strategy_opt_values,
                    cash_naive,b_real[i],T,i,y_opt_cash_naive)
            if cash_naive == -1.0E6:
                regret_naive_strategy = best_strategy_return
        B = get_B(b, params, T, predict_method)
        if regret_robust_strategy < best_strategy_return:
            robust_strategy_return, robust_strategy_opt_values =\
                robust_strategy(c, A, b, cash_robust, B, sigma_robust)
            cash_robust, y_opt_cash_robust = \
                update_cash(A, robust_strategy_opt_values,
                            cash_robust, b_real[i],T,i,y_opt_cash_robust)
            if cash_robust == -1.0E6:
                regret_robust_strategy = best_strategy_return
        if regret_affine_recourse < best_strategy_return:
            affine_recourse_return, affine_recourse_opt_values =\
                affine_recourse(c, A, b, cash_affine_recourse, B,
                                sigma_affine_recourse)
            cash_affine_recourse, y_opt_cash_affine_recourse = \
                update_cash(A,affine_recourse_opt_values,
                            cash_affine_recourse,b_real[i],
                            T,i,y_opt_cash_affine_recourse)
            if cash_affine_recourse == -1.0E6:
                regret_affine_recourse = best_strategy_return
        if regret_affine_cash < best_strategy_return:
            affine_cash_return, affine_cash_opt_values = \
                affine_cash(c, A, b, cash_affine_cash, B,
                                sigma_affine_recourse)
            cash_affine_cash, y_opt_cash_affine_cash = \
                update_cash(A, affine_cash_opt_values,
                            cash_affine_cash, b_real[i],
                            T, i, y_opt_cash_affine_cash)
            if cash_affine_cash == -1.0E6:
                regret_affine_cash = best_strategy_return
        if i <= 2:
            c = np.delete(c, [0, T - i - 1, T - 2 * i + 2])
            A = np.delete(A, 0, axis=0)
            A = np.delete(A, [0, T - i - 1, T - 2 * i + 2], axis=1)
        elif i<=4:
            c = np.delete(c,[0,5-i])
            A = np.delete(A, 0, axis=0)
            A = np.delete(A, [0,5-i], axis=1)
    if regret_naive_strategy == 0:
        regret_naive_strategy = best_strategy_return - cash_naive
    if regret_robust_strategy == 0:
        regret_robust_strategy = best_strategy_return - cash_robust
    if regret_affine_recourse == 0:
        regret_affine_recourse = best_strategy_return - cash_affine_recourse
    if regret_affine_cash == 0:
        regret_affine_cash = best_strategy_return - cash_affine_cash
    return best_strategy_return, regret_naive_strategy, \
           regret_robust_strategy, regret_affine_recourse,\
           regret_affine_cash

if __name__ == '__main__':
    all_liability_history = np.loadtxt('projectdata.txt')
    test_data = np.loadtxt('test_data.txt')

    predict_method = 'AR1'
    cash_naive = 70.3
    cash_robust = 70.3
    cash_affine_recourse = 70.3
    cash_affine_cash = 70.3
    sigma_robust = 3.0
    sigma_affine_recourse = 3.0
    T = 6
    c = np.append(np.zeros(13), 1)
    A = np.array([[1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                  [-1.01, 1, 0, 0, 0, 0, 1, 0, 1.003, -1, 0, 0, 0, 0],
                  [0, -1.01, 1, 0, 0, 0, 0, 1, 0, 1.003, -1, 0, 0, 0],
                  [0, 0, -1.01, 1, 0, -1.02, 0, 0, 0, 0, 1.003, -1, 0, 0],
                  [0, 0, 0, -1.01, 1, 0, -1.02, 0, 0, 0, 0, 1.003, -1, 0],
                  [0, 0, 0, 0, -1.01, 0, 0, -1.02, 0, 0, 0, 0, 1.003, -1]])

    ### Training on historical data
    dict_expected_regret = {"Naive strategy regret": 0,
                            "Robust strategy regret": 0,
                            "Affine Recourse regret": 0,
                            "Affine cash regret": 0}
    dict_best_strategy = {"Naive strategy": 0,
                      "Robust strategy": 0,
                      "Affine Recourse": 0,
                      "Affine cash": 0,
                      "None feasible": 0}
    count = 0
    for i in range(54, 6, -1):
        count += 1
        training_data = all_liability_history[:-i]
        b_real = all_liability_history[-i:-i + 6]
        best_strategy_return, regret_naive_strategy, \
        regret_robust_strategy, regret_affine_recourse, \
        regret_affine_cash = run(c, A,
        all_liability_history[-i:-i+6], T,cash_naive,cash_robust,cash_affine_recourse,
                                                             cash_affine_cash,
        all_liability_history[:-i],sigma_robust,sigma_affine_recourse,predict_method)
        dict_expected_regret["Naive strategy regret"] += regret_naive_strategy
        dict_expected_regret["Robust strategy regret"] += regret_robust_strategy
        dict_expected_regret["Affine Recourse regret"] += regret_affine_recourse
        dict_expected_regret["Affine cash regret"] += regret_affine_cash
        if regret_affine_recourse == regret_naive_strategy and\
            regret_affine_recourse == regret_robust_strategy:
            best_strategy_index = 4
        else:
            best_strategy_index = np.argmin([
                regret_naive_strategy,
                regret_robust_strategy,
                regret_affine_recourse])
        if best_strategy_index == 0:
            dict_best_strategy["Naive strategy"] += 1
        elif best_strategy_index == 1:
            dict_best_strategy["Robust strategy"] += 1
        elif best_strategy_index == 2:
            dict_best_strategy["Affine Recourse"] += 1
        elif best_strategy_index == 3:
            dict_best_strategy["Affine cash"] += 1
        elif best_strategy_index == 4:
            dict_best_strategy["None feasible"] += 1
    dict_expected_regret["Naive strategy regret"] /= count
    dict_expected_regret["Robust strategy regret"] /= count
    dict_expected_regret["Affine Recourse regret"] /= count
    dict_expected_regret["Affine cash regret"] /= count
    print dict_expected_regret
    print dict_best_strategy

    ### Test data
    training_data = all_liability_history
    b_real = test_data
    best_strategy_return, regret_naive_strategy, \
    regret_robust_strategy, regret_affine_recourse, \
    regret_affine_cash = \
                  run(c, A,
                  b_real, T,
                  cash_naive,
                  cash_robust,
                  cash_affine_recourse,
                      cash_affine_cash,
                  training_data,
                  sigma_robust,
                  sigma_affine_recourse,
                  predict_method)
    print "Regret for naive strategy = " + str(regret_naive_strategy)
    print "Regret for robust strategy = " + str(regret_robust_strategy)
    print "Regret for affine recourse = " + str(regret_affine_recourse)
    print "Regret for affine cash = " + str(regret_affine_cash)
    all_strategies = ["Naive Strategy", "Robust strategy",
                      "Affine Recourse", "Affine cash", "None feasible"]
    if regret_affine_recourse == regret_naive_strategy and \
                    regret_affine_recourse == regret_robust_strategy:
        best_strategy_index = 4
    else:
        best_strategy_index = np.argmin([regret_naive_strategy,
                    regret_robust_strategy, regret_affine_recourse])
    print "Best strategy is " + all_strategies[best_strategy_index]

    ### More test data
    dict_expected_regret = {"Naive strategy regret": 0,
                            "Robust strategy regret": 0,
                            "Affine Recourse regret": 0,
                            "Affine cash regret": 0}
    dict_best_strategy = {"Naive strategy": 0,
                      "Robust strategy": 0,
                      "Affine Recourse": 0,
                      "Affine cash": 0,
                      "None feasible": 0}
    lines = [line.rstrip('\n')[:-1] for line in open('more_test_data.txt')]
    count = 0
    for i in range(len(lines)):
        count += 1
        training_data = all_liability_history
        b_real = [float(j) for j in lines[i].split(',')]
        best_strategy_return, regret_naive_strategy, \
        regret_robust_strategy, regret_affine_recourse, \
        regret_affine_cash = \
            run(c, A,
                b_real, T,
                cash_naive,
                cash_robust,
                cash_affine_recourse,
                cash_affine_cash,
                training_data,
                sigma_robust,
                sigma_affine_recourse,
                predict_method)
        dict_expected_regret["Naive strategy regret"] += regret_naive_strategy
        dict_expected_regret["Robust strategy regret"] += regret_robust_strategy
        dict_expected_regret["Affine Recourse regret"] += regret_affine_recourse
        dict_expected_regret["Affine cash regret"] += regret_affine_cash
        if regret_affine_recourse == regret_naive_strategy and\
                        regret_affine_recourse == regret_robust_strategy:
            best_strategy_index = 4
        else:
            best_strategy_index = np.argmin([regret_naive_strategy,
                            regret_robust_strategy, regret_affine_recourse,
                                             regret_affine_cash])
        if best_strategy_index == 0:
            dict_best_strategy["Naive strategy"] += 1
        elif best_strategy_index == 1:
            dict_best_strategy["Robust strategy"] += 1
        elif best_strategy_index == 2:
            dict_best_strategy["Affine Recourse"] += 1
        elif best_strategy_index == 3:
            dict_best_strategy["Affine cash"] += 1
        elif best_strategy_index == 4:
            dict_best_strategy["None feasible"] += 1
    dict_expected_regret["Naive strategy regret"] /= count
    dict_expected_regret["Robust strategy regret"] /= count
    dict_expected_regret["Affine Recourse regret"] /= count
    dict_expected_regret["Affine cash regret"] /= count
    print dict_expected_regret
    print dict_best_strategy