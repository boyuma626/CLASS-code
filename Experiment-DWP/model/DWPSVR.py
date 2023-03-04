import numpy as np
# from coptpy import *
import cvxopt
import cvxpy as cp

# 向量的交叉项表示
# 输入的是矩阵N*n维，N是样本数，n是特征数
def lvec(x):
    N, n = x.shape
    s = []
    for i in range(N):
        for j in range(n):
            for k in range(j, n):
                if k == j:
                    c = 1 / 2 * x[i, j] * x[i, j]
                    s.append(c)
                else:
                    c = x[i, j] * x[i, k]
                    s.append(c)

    s = np.array(s).reshape(N, -1)

    return s


# 返回的也是N*m矩阵，m对应的是升维后列数

# 产生G矩阵
def generateG(x):
    N, n = x.shape
    e = np.ones((N, 1))
    s = lvec(x)
    z = np.hstack((s, x, e))
    eta = lvec(z)
    r = np.hstack((eta, s))
    G = np.hstack((r, x, e))
    return G


class DWPSVR:
    """
    :parameter
    C1:
    C2:
    nu1:
    lambda1:
    """
    def __init__(self, C1, C2, nu1, lambda1):
        self.C1 = C1
        self.C2 = C2
        self.nu1 = nu1
        self.lambda1 = lambda1
        self.u1 = None
        self.u2 = None
        self.rmse_tr = None
        self.mape_tr = None
        self.rmse_te = None
        self.mape_te = None
        self.x_tr = None
        self.y_tr = None
        self.y_tr_predict = None
        self.x_te = None
        self.y_te = None
        self.y_te_predict = None

    def fit(self, x_tr, y_tr):#应该可以用输入x替代G
        C1 = self.C1
        C2 = self.C2
        nu1 = self.nu1
        lambda1 = self.lambda1
        
        G_tr=generateG(x_tr)

        N, l = G_tr.shape
        
        e = np.ones((N, 1))
        alpha1 = cp.Variable((N, 1))
        r1 = cp.Variable((N, 1))
        r11 = cp.Variable((N, 1))
        expr1 = 1 / (2 * lambda1)
        expr2 = lambda1 * y_tr - (alpha1 + (r11 - r1))
        expr3 = G_tr @ np.linalg.inv(G_tr.T @ G_tr + C1 * np.identity(l, dtype=int)) @ G_tr.T
        expr3 = cp.atoms.affine.wraps.psd_wrap(expr3)
        expr4 = alpha1 + (r11 - r1)
        q = y_tr

        objective = cp.Minimize((expr1) * cp.quad_form(expr2, expr3) + q.T @ expr4)
        constraints = [0 <= alpha1, alpha1 <= C2 * e / N, r1 + r11 <= 1 - lambda1, 0 <= sum(alpha1),
                       sum(alpha1) <= C2 * nu1, 0 <= r1, 0 <= r11]
        prob = cp.Problem(objective, constraints)
        # results = prob.solve(solver=cp.GUROBI, verbose=True)
        results = prob.solve(solver='COPT')

        obj_val = []
        obj_val.append(prob.objective.value)

        # 求解TWDWPSVR-2对偶问题

        # 指明参数有助于调节参数以达到最优
        C3 = self.C1
        C4 = self.C2
        nu2 = self.nu1
        lambda2 = self.lambda1

        e = np.ones((N, 1))

        r2 = cp.Variable((N, 1))
        r21 = cp.Variable((N, 1))
        alpha2 = cp.Variable((N, 1))

        expr11 = 1 / (2 * lambda2)
        expr22 = lambda2 * y_tr + (alpha2 + (r2 - r21))
        # expr2 = (lambda1*y-alpha1+e*(r-r1)).T
        expr33 = G_tr @ np.linalg.inv(G_tr.T @ G_tr + C3 * np.identity(l, dtype=int)) @ G_tr.T
        expr33 = cp.atoms.affine.wraps.psd_wrap(expr33)
        expr44 = alpha2 + (r2 - r21)
        q = -y_tr

        objective = cp.Minimize((expr11) * cp.quad_form(expr22, expr33) + q.T @ expr44)
        # 定义约束条件
        constraints = [0 <= alpha2,
                       alpha2 <= C4 * e / N,
                       r2 + r21 <= 1 - lambda2,
                       0 <= sum(alpha2),
                       sum(alpha2) <= C4 * nu2,
                       0 <= r2, 0 <= r21]

        # 建⽴模型
        prob = cp.Problem(objective, constraints)
        # 模型求解
        # results = prob.solve(solver=cp.GLPK_MI, verbose=True)
        # results = prob.solve(solver=cp.GUROBI, verbose=True)
        results = prob.solve(solver='COPT')
        # print('问题的最优值为：{:.0f}'.format(prob.objective.value))
        a=expr2.value
        u1=1/lambda1*np.linalg.inv(G_tr.T@G_tr+C1*np.identity(l, dtype=int))@G_tr.T@a
        b=expr22.value
        u2=1/lambda2*np.linalg.inv(G_tr.T@G_tr+C3*np.identity(l, dtype=int))@G_tr.T@b
        
        self.u1 = u1
        self.u2 = u2
        
        # 计算error
        rmse_tr = 0
        mape_tr = 0
        self.rmse_tr = rmse_tr
        self.mape_tr = mape_tr
        
        self.x_tr = x_tr
        self.y_tr = y_tr
        
    def predict(self, x_te, y_te): #仅需要x_tr去训练
        G_te= generateG(x_te)
        u1 = self.u1
        u2 = self.u2
        y_hat1=G_te@u1
        y_hat2=G_te@u2
        y_hat=(y_hat1+y_hat2)/2
        self.y_te_predict = y_hat
        
        rmse_te = 0
        self.rmse_te = rmse_te
        mape_te = 0
        self.mape_te = mape_te
        
        self.x_te = x_te
        self.y_te = y_te

    

def GridSearch(model, parameter, data):
    models = []
    for C1 in parameter:
        svr = DWPSVR(C1=C1)
        svr.fit(data[0], data[1])
        models.append(svr)
    metric = []
    for i in range(len(models)):
        metric.append(models[i].rmse_te)
    
    k = np.argmin(metric)
    optimal_svr = models[k]
    optimal_parameter = optimal_svr.C1
    return optimal_svr


class QSSVR:
    pass