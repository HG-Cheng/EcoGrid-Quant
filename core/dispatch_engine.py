import numpy as np
from scipy.optimize import minimize


class EcoGridOptimizer:
    """
    EcoGrid 绿色电网调度优化引擎
    使用 SciPy SLSQP (序列最小二乘规划) 求解器
    """

    def __init__(self, carbon_tax_rate=50.0, bess_capacity=100.0):
        # --- V1 遗留参数：财务成本与碳排 ---
        self.carbon_tax_rate = carbon_tax_rate
        self.coal_cost = 30.0          # $/MWh 火电单位发电成本
        self.coal_emission = 0.8       # 吨 CO2/MWh 火电单位碳排强度

        # --- V2 & V3 核心：BESS (巨型电池储能系统) 物理与热力学参数 ---
        self.bess_capacity = bess_capacity  # MWh 电池最大储能容量 <--- 解除封印！由外部传入

        # 为了更贴近现实，最大功率通常与容量有“充放电倍率 (C-rate)”的关系
        # 假设我们使用的是 4 小时储能系统 (0.25C)，即最大功率 = 容量 / 4
        self.bess_max_power = bess_capacity / 4.0  # MW 最大充/放电功率（决定套利的速度）
        self.eff_charge = 0.95         # 充电效率（跨期能量流转的热力学折损）
        self.eff_discharge = 0.95      # 放电效率
        self.initial_soc = 0.0         # 初始电量状态 (State of Charge, MWh)

        print(
            f"⚙️ EcoGrid 优化引擎已初始化！碳税: {self.carbon_tax_rate} €/t | "
            f"电池容量: {self.bess_capacity} MWh"
        )

    def check_engine_status(self):
        """打印引擎当前的核心参数配置，用于快速自检"""
        print("=" * 40)
        print("🔍 EcoGrid 引擎状态自检")
        print(f"  碳税税率      : {self.carbon_tax_rate} €/t")
        print(f"  火电成本      : {self.coal_cost} $/MWh")
        print(f"  火电碳排强度  : {self.coal_emission} 吨CO2/MWh")
        print(f"  电池容量上限  : {self.bess_capacity} MWh")
        print(f"  最大充放功率  : {self.bess_max_power} MW")
        print(f"  充/放电效率   : {self.eff_charge} / {self.eff_discharge}")
        print(f"  初始电量(SoC) : {self.initial_soc} MWh")
        print("=" * 40)
        print("✅ 引擎状态正常，已就绪。")

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
        result = minimize(
            objective_cost, x0, method='SLSQP', bounds=bounds, constraints=constraints
        )

        return result

    def optimize_horizon(self, demand_series, wind_avail, solar_avail, horizon=24):
        """
        V2 核心引擎：跨期动态调度 (基于多期非线性规划 NLP / MPC 思想)
        入参 series 应当是长度等于 horizon 的 numpy 数组。
        每个时间步有 6 个决策变量: [火电, 风电, 光伏, 充电, 放电, SoC(当前电量)]
        因此 SciPy 需要同时求解一个长度为 6 * horizon 的一维数组 x。
        """
        vars_per_step = 6
        n_vars = vars_per_step * horizon

        # --- 1. 生成多期物理边界 (Bounds) ---
        # 用循环将未来 H 个小时的物理限制展平送给 SciPy
        bounds = []
        for t in range(horizon):
            bounds.extend([
                (0, None),                  # x[6t + 0]: P_coal (火电兜底，无上限)
                (0, wind_avail[t]),         # x[6t + 1]: P_wind (风电出力不能超自然禀赋)
                (0, solar_avail[t]),        # x[6t + 2]: P_solar(光伏出力不能超自然禀赋)
                (0, self.bess_max_power),   # x[6t + 3]: P_charge (充电功率限制)
                (0, self.bess_max_power),   # x[6t + 4]: P_discharge (放电功率限制)
                (0, self.bess_capacity)     # x[6t + 5]: SoC (不能把电池充爆)
            ])

        # --- 2. 定义全局目标函数 (Objective Function) - 跨期总成本极小化 ---
        def objective(x):
            total_cost = 0.0
            for t in range(horizon):
                idx = t * vars_per_step
                p_coal = x[idx + 0]
                # 只有火电产生真实的经济与环境双重成本，风光和充放电的边际成本视为 0
                step_cost = p_coal * (self.coal_cost + self.coal_emission * self.carbon_tax_rate)
                total_cost += step_cost
            return total_cost

        # --- 3. 定义严格的物理系统约束 (Constraints) ---
        constraints = []

        # 针对未来预测窗口的每一小时 t，生成两道“物理枷锁”
        for t in range(horizon):

            # ---------------------------------------------------------
            # 约束 1：供需绝对平衡 (Power Balance)
            # 公式：总供给 (火+风+光+放电) - 总消耗 (负荷需求+充电) == 0
            # ---------------------------------------------------------
            # 极客细节：使用工厂函数 make_balance_constraint(step)
            # 这是为了防止 Python 经典的 "循环变量延迟绑定 (Late Binding)" 陷阱
            def make_balance_constraint(step):
                def balance_rule(x):
                    idx = step * vars_per_step
                    p_coal = x[idx + 0]
                    p_wind = x[idx + 1]
                    p_solar = x[idx + 2]
                    p_charge = x[idx + 3]
                    p_discharge = x[idx + 4]

                    total_supply = p_coal + p_wind + p_solar + p_discharge
                    total_demand = demand_series[step] + p_charge
                    return total_supply - total_demand
                return balance_rule

            constraints.append({'type': 'eq', 'fun': make_balance_constraint(t)})

            # ---------------------------------------------------------
            # 约束 2：电量状态转移方程 (SoC Dynamics)
            # 公式：当前电量 - (上一期电量 + 实际充入电池的电量 - 实际抽干电池的电量) == 0
            # ---------------------------------------------------------
            def make_soc_constraint(step):
                def soc_rule(x):
                    idx = step * vars_per_step
                    p_charge = x[idx + 3]
                    p_discharge = x[idx + 4]
                    soc_current = x[idx + 5]

                    # 严谨的热力学损耗计算
                    actual_charge = p_charge * self.eff_charge
                    actual_drain = p_discharge / self.eff_discharge

                    # 时间耦合：如果是第 0 小时，看初始电量；否则看上一个小时求解器决定的电量
                    soc_prev = self.initial_soc if step == 0 else x[(step - 1) * vars_per_step + 5]

                    return soc_current - (soc_prev + actual_charge - actual_drain)
                return soc_rule

            constraints.append({'type': 'eq', 'fun': make_soc_constraint(t)})

        # --- 4. 组装并调用求解器 ---
        # 初始猜测值 (Initial Guess)：144 个变量全部先盲猜为 0
        x0 = np.zeros(n_vars)

        # 运行非线性规划求解器 (NLP)
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result
