import random
import numpy as np
import copy
from typing import Dict, Optional, List, Any, Tuple
from .scenario_context import ScenarioContext


class ScenarioGenerator:
    """
    电力网络场景生成器
    用于生成多样化的训练场景，提高智能体的鲁棒性
    """

    def __init__(self, base_case: Dict, seed: Optional[int] = None, config: Optional[Dict] = None):
        """
        初始化场景生成器

        Args:
            base_case: 基础案例数据（MATPOWER格式）
            seed: 随机种子
            config: 配置字典，用于控制输出详细程度
        """
        self.base_case = base_case
        self.config = config

        # 获取调试配置
        debug_config = config.get('debug', {}) if config else {}
        self.training_output = debug_config.get('training_output', {})

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_scene(self, 
                            perturb_types: Optional[List[str]] = None,
                            perturb_prob: float = 0.8) -> Tuple[Dict, ScenarioContext]:
        """
        生成随机扰动的电力网络场景
        
        Args:
            perturb_types: 允许的扰动类型列表，默认为所有类型
            perturb_prob: 应用扰动的概率
            
        Returns:
            Tuple[Dict, ScenarioContext]: (扰动后的案例数据, 场景上下文)
        """
        # 深拷贝以避免修改原始数据
        perturbed_case = copy.deepcopy(self.base_case)
        
        # 初始化场景上下文
        scenario_context = ScenarioContext()
        
        # 判断是否应用扰动
        if random.random() > perturb_prob:
            return perturbed_case, scenario_context
        
        # 可用的扰动类型
        available_types = perturb_types or ['n-1', 'load_gen_fluctuation', 'both', 'none']
        perturb_type = random.choice(available_types)
        
        # 应用扰动并记录上下文
        if perturb_type in ['n-1', 'both']:
            fault_branch_idx = self._apply_n1_contingency(perturbed_case)
            scenario_context.has_n1_fault = True
            scenario_context.fault_branch_idx = fault_branch_idx
            
        if perturb_type in ['load_gen_fluctuation', 'both']:
            scale_factor = self._apply_injection_perturbation(perturbed_case)
            scenario_context.load_scale_factor = scale_factor
            scenario_context.has_gen_fluctuation = True
            
        return perturbed_case, scenario_context
    
    def _apply_n1_contingency(self, case_data: Dict) -> Optional[int]:
        """
        应用N-1故障（随机断开一条线路）
        
        Args:
            case_data: 要修改的案例数据
            
        Returns:
            断开的线路索引，如果没有线路可断开则返回None
        """
        if 'branch' not in case_data:
            return None
            
        all_branches = case_data['branch']
        # 找到所有活跃的线路（状态为1）
        active_indices = np.where(all_branches[:, 10] == 1)[0]
        
        if len(active_indices) > 0:
            # 随机选择一条线路断开
            idx = random.choice(active_indices)
            all_branches[idx, 10] = 0  # 设置线路状态为0（断开）

            # 根据配置决定是否显示详细信息
            show_scenario_generation = self.training_output.get('show_scenario_generation', True)
            only_show_errors = self.training_output.get('only_show_errors', False)

            if show_scenario_generation and not only_show_errors:
                print(f"🔧 N-1故障：断开线路 {idx} (从母线 {int(all_branches[idx, 0])} 到 {int(all_branches[idx, 1])})")
            
            return idx
        
        return None
    
    def _apply_injection_perturbation(self, case_data: Dict, 
                                    scale_range: tuple = (0.8, 1.2)) -> float:
        """
        应用注入功率扰动（负荷和发电机波动）
        
        Args:
            case_data: 要修改的案例数据
            scale_range: 缩放范围
            
        Returns:
            实际使用的缩放因子
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

        # 根据配置决定是否显示详细信息
        show_scenario_generation = self.training_output.get('show_scenario_generation', True)
        only_show_errors = self.training_output.get('only_show_errors', False)

        if show_scenario_generation and not only_show_errors:
            print(f"🔧 注入扰动：缩放因子 = {scale:.3f}")
        
        return scale
    
    def generate_batch_scenarios(self, 
                               num_scenarios: int,
                               perturb_types: Optional[List[str]] = None) -> List[Tuple[Dict, ScenarioContext]]:
        """
        批量生成多个场景
        
        Args:
            num_scenarios: 要生成的场景数量
            perturb_types: 允许的扰动类型
            
        Returns:
            场景和上下文的元组列表
        """
        scenarios = []
        for i in range(num_scenarios):
            scenario_data, scenario_context = self.generate_random_scene(perturb_types)
            scenarios.append((scenario_data, scenario_context))
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