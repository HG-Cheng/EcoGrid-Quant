import numpy as np
from scipy.optimize import minimize

class EcoGridOptimizer:
    """
    EcoGrid 绿色电网调度优化引擎 
    使用 SciPy SLSQP (序列最小二乘规划) 求解器
    """
    
    def __init__(self, carbon_tax_rate=50.0):
        self.carbon_tax_rate = carbon_tax_rate
        print(f"⚙️ EcoGrid 优化引擎已初始化！当前碳税设定: {self.carbon_tax_rate} €/t")
        
    def optimize_dispatch(self, demand, wind_avail, solar_avail):
        """
        核心调度算法
        :param demand: 当前电网总需求 (MW)
        :param wind_avail: 当前风电最大可用量 (MW)
        :param solar_avail: 当前光伏最大可用量 (MW)
        """
        # --- 1. 定义目标函数 (我们极力想要变小的值：总成本) ---
        # 变量 x 的含义: x[0]是风电出力, x[1]是光伏出力, x[2]是火电出力
        def objective_cost(x):
            # 发电边际成本 (假设：风电2欧/MW，光伏1.5欧/MW，火电40欧/MW)
            generation_cost = x[0]*2.0 + x[1]*1.5 + x[2]*40.0
            
            # 碳排放惩罚 (假设：火电每发1MW产生0.9吨碳排)
            carbon_emission = x[2] * 0.9 
            carbon_penalty = carbon_emission * self.carbon_tax_rate
            
            return generation_cost + carbon_penalty

        # --- 2. 定义物理约束条件 (必须遵守的铁律) ---
        # 约束：三种电加起来，必须等于总需求 demand (即 x[0]+x[1]+x[2] - demand == 0)
        constraints = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - demand})

        # --- 3. 定义物理边界 (Bounds) ---
        # 风电不能小于0，不能大于当前可用量；光伏同理；火电假设最大容量为1000MW
        bounds = ((0, wind_avail), (0, solar_avail), (0, 1000))

        # --- 4. 初始猜测值 (随便给个起点，让求解器开始迭代) ---
        x0 = [0, 0, demand] # 假设一开始全是火电

        # --- 5. 🚀 启动 SciPy 求解器！ ---
        # 使用 SLSQP 算法寻找最优解
        result = minimize(objective_cost, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result