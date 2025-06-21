import random
import numpy as np
import copy
from typing import Dict, Optional, List


class ScenarioGenerator:
    """
    电力网络场景生成器
    用于生成多样化的训练场景，提高智能体的鲁棒性
    """
    
    def __init__(self, base_case: Dict, seed: Optional[int] = None):
        """
        初始化场景生成器
        
        Args:
            base_case: 基础案例数据（MATPOWER格式）
            seed: 随机种子
        """
        self.base_case = base_case
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_scene(self, 
                            perturb_types: Optional[List[str]] = None,
                            perturb_prob: float = 0.8) -> Dict:
        """
        生成随机扰动的电力网络场景
        
        Args:
            perturb_types: 允许的扰动类型列表，默认为所有类型
            perturb_prob: 应用扰动的概率
            
        Returns:
            扰动后的案例数据
        """
        # 深拷贝以避免修改原始数据
        perturbed_case = copy.deepcopy(self.base_case)
        
        # 判断是否应用扰动
        if random.random() > perturb_prob:
            return perturbed_case
        
        # 可用的扰动类型
        available_types = perturb_types or ['n-1', 'load_gen_fluctuation', 'both', 'none']
        perturb_type = random.choice(available_types)
        
        # 应用扰动
        if perturb_type in ['n-1', 'both']:
            self._apply_n1_contingency(perturbed_case)
            
        if perturb_type in ['load_gen_fluctuation', 'both']:
            self._apply_injection_perturbation(perturbed_case)
            
        return perturbed_case
    
    def _apply_n1_contingency(self, case_data: Dict):
        """
        应用N-1故障（随机断开一条线路）
        
        Args:
            case_data: 要修改的案例数据
        """
        if 'branch' not in case_data:
            return
            
        all_branches = case_data['branch']
        # 找到所有活跃的线路（状态为1）
        active_indices = np.where(all_branches[:, 10] == 1)[0]
        
        if len(active_indices) > 0:
            # 随机选择一条线路断开
            idx = random.choice(active_indices)
            all_branches[idx, 10] = 0  # 设置线路状态为0（断开）
            print(f"🔧 N-1故障：断开线路 {idx} (从母线 {int(all_branches[idx, 0])} 到 {int(all_branches[idx, 1])})")
    
    def _apply_injection_perturbation(self, case_data: Dict, 
                                    scale_range: tuple = (0.8, 1.2)):
        """
        应用注入功率扰动（负荷和发电机波动）
        
        Args:
            case_data: 要修改的案例数据
            scale_range: 缩放范围
        """
        scale = random.uniform(*scale_range)
        
        # 扰动负荷
        if 'bus' in case_data:
            # PD (有功负荷) 在第3列，QD (无功负荷) 在第4列
            case_data['bus'][:, 2] *= scale
            case_data['bus'][:, 3] *= scale
            
        # 扰动发电机
        if 'gen' in case_data:
            # PG (有功发电) 在第2列，Pmax在第9列
            orig_pg = case_data['gen'][:, 1].copy()
            pmax = case_data['gen'][:, 8]
            # 确保不超过Pmax限制
            case_data['gen'][:, 1] = np.minimum(orig_pg * scale, pmax)
            
        print(f"🔧 注入扰动：缩放因子 = {scale:.3f}")
    
    def generate_batch_scenarios(self, 
                               num_scenarios: int,
                               perturb_types: Optional[List[str]] = None) -> List[Dict]:
        """
        批量生成多个场景
        
        Args:
            num_scenarios: 要生成的场景数量
            perturb_types: 允许的扰动类型
            
        Returns:
            场景列表
        """
        scenarios = []
        for i in range(num_scenarios):
            scenario = self.generate_random_scene(perturb_types)
            scenarios.append(scenario)
        return scenarios
    
    def apply_specific_contingency(self, 
                                 case_data: Dict,
                                 branch_idx: int) -> Dict:
        """
        应用特定的N-1故障
        
        Args:
            case_data: 基础案例数据
            branch_idx: 要断开的线路索引
            
        Returns:
            应用故障后的案例
        """
        perturbed_case = copy.deepcopy(case_data)
        if 'branch' in perturbed_case and 0 <= branch_idx < len(perturbed_case['branch']):
            perturbed_case['branch'][branch_idx, 10] = 0
        return perturbed_case
    
    def apply_load_scaling(self, 
                          case_data: Dict,
                          bus_indices: Optional[List[int]] = None,
                          scale_factor: float = 1.0) -> Dict:
        """
        对特定母线应用负荷缩放
        
        Args:
            case_data: 基础案例数据
            bus_indices: 要缩放的母线索引列表，None表示所有母线
            scale_factor: 缩放因子
            
        Returns:
            缩放后的案例
        """
        perturbed_case = copy.deepcopy(case_data)
        if 'bus' not in perturbed_case:
            return perturbed_case
            
        if bus_indices is None:
            bus_indices = range(len(perturbed_case['bus']))
            
        for idx in bus_indices:
            if 0 <= idx < len(perturbed_case['bus']):
                perturbed_case['bus'][idx, 2] *= scale_factor  # PD
                perturbed_case['bus'][idx, 3] *= scale_factor  # QD
                
        return perturbed_case 