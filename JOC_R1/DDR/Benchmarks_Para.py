class SimulatorConfig:
    def __init__(self):
        self.seed = 3
        # 迭代次数
        self.iters = 100
        
        # 参数 p 和 d
        self.p = 4
        self.d = 10
        
        # 样本数量
        self.samples_test = 10000
        self.samples_train = 100
        
        # 输入变量的上下界
        self.lower = 0
        self.upper = 1
        
        # 其他数值参数
        self.alpha = 1
        self.n_epsilon = 1
        self.mis = 1
        self.thres = 10000
        self.ver = 1
        
        # 分布设置
        self.x_dister = 'uniform'   # 输入变量分布类型
        self.e_dister = 'normal'    # 噪声项分布类型
        
        # 输入变量分布参数
        self.xl = -2                # uniform 分布下限
        self.xu = 2                 # uniform 分布上限
        self.xm = 2                 # normal 分布均值（可能用于噪声）
        self.xv = 0.25              # normal 分布方差
        
        # 模型相关参数
        self.bp = 7                 # 可能是 breakpoint 或其他用途
        self.mu = 0.25              # 可调参数 μ
        self.lamb = 0.25            # lambda 正则化参数或其他用途

    def __str__(self):
        """返回配置的字符串表示"""
        return "\n".join([f"{key} = {value}" for key, value in self.__dict__.items()])

