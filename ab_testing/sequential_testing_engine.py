import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import warnings
from math import log, sqrt, exp


class SequentialTestType(Enum):
    SPRT = "sequential_probability_ratio_test"
    GROUP_SEQUENTIAL = "group_sequential"
    ALWAYS_VALID = "always_valid"
    ALPHA_SPENDING = "alpha_spending"


class SequentialBoundary(Enum):
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    HAYBITTLE_PETO = "haybittle_peto"
    CUSTOM = "custom"


@dataclass
class SequentialDesign:
    test_type: SequentialTestType
    boundary_type: SequentialBoundary
    alpha: float
    beta: float
    max_sample_size: int
    interim_analyses: List[float]
    theta_0: float  # Null hypothesis value
    theta_1: float  # Alternative hypothesis value


@dataclass
class SequentialResult:
    analysis_time: datetime
    interim_number: int
    sample_size: int
    test_statistic: float
    p_value: float
    lower_bound: float
    upper_bound: float
    crossed_boundary: bool
    decision: str
    confidence_interval: Tuple[float, float]
    power: float


class SequentialTestingEngine:
    """Sequential testing engine with multiple stopping rules and boundaries"""
    
    def __init__(self):
        self.designs: Dict[str, SequentialDesign] = {}
        self.test_histories: Dict[str, List[SequentialResult]] = {}
        self.active_tests: Dict[str, Dict] = {}
        
    def create_sequential_test(
        self,
        test_id: str,
        test_type: SequentialTestType = SequentialTestType.GROUP_SEQUENTIAL,
        boundary_type: SequentialBoundary = SequentialBoundary.OBRIEN_FLEMING,
        alpha: float = 0.05,
        beta: float = 0.2,
        max_sample_size: int = 10000,
        effect_size: float = 0.05,
        interim_analyses: Optional[List[float]] = None
    ) -> str:
        """Create a sequential testing design"""
        
        if interim_analyses is None:
            # Default to 5 equally spaced interim analyses
            interim_analyses = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Calculate null and alternative hypotheses
        theta_0 = 0.0  # No difference
        theta_1 = effect_size  # Expected effect size
        
        design = SequentialDesign(
            test_type=test_type,
            boundary_type=boundary_type,
            alpha=alpha,
            beta=beta,
            max_sample_size=max_sample_size,
            interim_analyses=interim_analyses,
            theta_0=theta_0,
            theta_1=theta_1
        )
        
        self.designs[test_id] = design
        self.test_histories[test_id] = []
        self.active_tests[test_id] = {
            'created_at': datetime.now(),
            'status': 'active',
            'cumulative_alpha_spent': 0.0,
            'cumulative_beta_spent': 0.0
        }
        
        return test_id
    
    def analyze_sequential(
        self,
        test_id: str,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]],
        analysis_time: Optional[datetime] = None
    ) -> SequentialResult:
        """Perform sequential analysis at interim point"""
        
        if test_id not in self.designs:
            raise ValueError(f"Test {test_id} not found")
        
        design = self.designs[test_id]
        test_info = self.active_tests[test_id]
        analysis_time = analysis_time or datetime.now()
        
        # Calculate current sample size and test statistic
        if 'conversions' in control_data and 'impressions' in control_data:
            test_stat, sample_size = self._calculate_proportion_test_statistic(
                control_data, treatment_data
            )
        else:
            test_stat, sample_size = self._calculate_continuous_test_statistic(
                control_data, treatment_data
            )
        
        # Determine information fraction
        info_fraction = sample_size / design.max_sample_size
        
        # Calculate boundaries based on test type
        if design.test_type == SequentialTestType.SPRT:
            result = self._analyze_sprt(design, test_stat, sample_size, analysis_time)
        elif design.test_type == SequentialTestType.GROUP_SEQUENTIAL:
            result = self._analyze_group_sequential(
                design, test_stat, sample_size, info_fraction, analysis_time
            )
        elif design.test_type == SequentialTestType.ALWAYS_VALID:
            result = self._analyze_always_valid(design, test_stat, sample_size, analysis_time)
        else:
            result = self._analyze_alpha_spending(
                design, test_stat, sample_size, info_fraction, analysis_time, test_info
            )
        
        # Store result
        self.test_histories[test_id].append(result)
        
        # Update test status if stopped
        if result.decision != 'continue':
            self.active_tests[test_id]['status'] = 'stopped'
            self.active_tests[test_id]['stopped_at'] = analysis_time
        
        return result
    
    def _calculate_proportion_test_statistic(
        self,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]]
    ) -> Tuple[float, int]:
        """Calculate test statistic for proportion data"""
        
        n1 = control_data['impressions']
        x1 = control_data['conversions']
        n2 = treatment_data['impressions']
        x2 = treatment_data['conversions']
        
        if n1 == 0 or n2 == 0:
            return 0.0, 0
        
        # Calculate proportions
        p1 = x1 / n1
        p2 = x2 / n2
        
        # Pooled proportion
        p_pooled = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Z-statistic
        if se == 0:
            z_stat = 0.0
        else:
            z_stat = (p2 - p1) / se
        
        total_sample = n1 + n2
        
        return z_stat, total_sample
    
    def _calculate_continuous_test_statistic(
        self,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]]
    ) -> Tuple[float, int]:
        """Calculate test statistic for continuous data"""
        
        n1 = control_data['sample_size']
        mean1 = control_data['mean']
        std1 = control_data['std']
        
        n2 = treatment_data['sample_size']
        mean2 = treatment_data['mean']
        std2 = treatment_data['std']
        
        if n1 == 0 or n2 == 0:
            return 0.0, 0
        
        # Pooled variance
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        
        # Standard error
        se = sqrt(pooled_var * (1/n1 + 1/n2))
        
        # T-statistic
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean2 - mean1) / se
        
        total_sample = n1 + n2
        
        return t_stat, total_sample
    
    def _analyze_sprt(
        self,
        design: SequentialDesign,
        test_stat: float,
        sample_size: int,
        analysis_time: datetime
    ) -> SequentialResult:
        """Sequential Probability Ratio Test analysis"""
        
        # SPRT boundaries
        log_alpha = log(design.alpha / (1 - design.beta))
        log_beta = log((1 - design.alpha) / design.beta)
        
        # Calculate log likelihood ratio
        # Simplified for normal case
        delta = design.theta_1 - design.theta_0
        log_lr = delta * test_stat - (delta**2 * sample_size) / 8
        
        # Decision boundaries
        if log_lr >= log_alpha:
            decision = "reject_null"
            crossed_boundary = True
        elif log_lr <= log_beta:
            decision = "accept_null" 
            crossed_boundary = True
        else:
            decision = "continue"
            crossed_boundary = False
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        # Confidence interval (approximate)
        ci_margin = 1.96 / sqrt(sample_size)
        effect_estimate = test_stat / sqrt(sample_size / 4)  # Approximate
        
        return SequentialResult(
            analysis_time=analysis_time,
            interim_number=len(self.test_histories.get(design.test_type.value, [])) + 1,
            sample_size=sample_size,
            test_statistic=test_stat,
            p_value=p_value,
            lower_bound=log_beta,
            upper_bound=log_alpha,
            crossed_boundary=crossed_boundary,
            decision=decision,
            confidence_interval=(effect_estimate - ci_margin, effect_estimate + ci_margin),
            power=1 - design.beta if decision == "reject_null" else design.beta
        )
    
    def _analyze_group_sequential(
        self,
        design: SequentialDesign,
        test_stat: float,
        sample_size: int,
        info_fraction: float,
        analysis_time: datetime
    ) -> SequentialResult:
        """Group sequential design analysis"""
        
        # Calculate boundaries based on boundary type
        if design.boundary_type == SequentialBoundary.OBRIEN_FLEMING:
            boundary = self._obrien_fleming_boundary(info_fraction, design.alpha)
        elif design.boundary_type == SequentialBoundary.POCOCK:
            boundary = self._pocock_boundary(info_fraction, design.alpha)
        else:
            boundary = stats.norm.ppf(1 - design.alpha / 2)  # Fixed boundary
        
        # Futility boundary (simple)
        futility_boundary = -boundary * 0.5  # Conservative futility
        
        # Decision logic
        if abs(test_stat) >= boundary:
            decision = "reject_null" if test_stat > 0 else "reject_null_negative"
            crossed_boundary = True
        elif test_stat < futility_boundary and info_fraction > 0.5:
            decision = "accept_null"
            crossed_boundary = True
        else:
            decision = "continue"
            crossed_boundary = False
        
        # Calculate p-value adjusted for multiple testing
        adjusted_p = self._calculate_adjusted_p_value(test_stat, info_fraction, design)
        
        # Confidence interval
        ci_width = boundary / sqrt(sample_size / 4)
        effect_estimate = test_stat / sqrt(sample_size / 4)
        
        return SequentialResult(
            analysis_time=analysis_time,
            interim_number=len(self.test_histories.get('group_sequential', [])) + 1,
            sample_size=sample_size,
            test_statistic=test_stat,
            p_value=adjusted_p,
            lower_bound=-boundary,
            upper_bound=boundary,
            crossed_boundary=crossed_boundary,
            decision=decision,
            confidence_interval=(effect_estimate - ci_width, effect_estimate + ci_width),
            power=self._calculate_conditional_power(test_stat, info_fraction, design)
        )
    
    def _analyze_always_valid(
        self,
        design: SequentialDesign,
        test_stat: float,
        sample_size: int,
        analysis_time: datetime
    ) -> SequentialResult:
        """Always valid inference analysis"""
        
        # Always valid confidence sequence
        # Using mixture approach with time-uniform confidence
        log_n = log(max(sample_size, 1))
        
        # Confidence sequence boundary
        boundary_factor = sqrt(2 * log_n + 4 * log(log_n) + 2 * log(1 / design.alpha))
        boundary = boundary_factor / sqrt(sample_size)
        
        # Effect estimate
        effect_estimate = test_stat / sqrt(sample_size / 4)
        
        # Decision based on confidence sequence
        if abs(effect_estimate) > boundary:
            decision = "reject_null"
            crossed_boundary = True
        else:
            decision = "continue"
            crossed_boundary = False
        
        # Always valid p-value (conservative)
        always_valid_p = min(1.0, design.alpha * exp(-test_stat**2 / 2))
        
        return SequentialResult(
            analysis_time=analysis_time,
            interim_number=len(self.test_histories.get('always_valid', [])) + 1,
            sample_size=sample_size,
            test_statistic=test_stat,
            p_value=always_valid_p,
            lower_bound=-boundary,
            upper_bound=boundary,
            crossed_boundary=crossed_boundary,
            decision=decision,
            confidence_interval=(effect_estimate - boundary, effect_estimate + boundary),
            power=1 - design.beta if decision == "reject_null" else design.beta
        )
    
    def _analyze_alpha_spending(
        self,
        design: SequentialDesign,
        test_stat: float,
        sample_size: int,
        info_fraction: float,
        analysis_time: datetime,
        test_info: Dict
    ) -> SequentialResult:
        """Alpha spending function approach"""
        
        # O'Brien-Fleming alpha spending
        alpha_spent = self._obrien_fleming_alpha_spending(info_fraction, design.alpha)
        
        # Incremental alpha
        incremental_alpha = alpha_spent - test_info['cumulative_alpha_spent']
        test_info['cumulative_alpha_spent'] = alpha_spent
        
        # Boundary calculation
        if incremental_alpha <= 0:
            boundary = float('inf')
        else:
            boundary = stats.norm.ppf(1 - incremental_alpha / 2)
        
        # Decision logic
        if abs(test_stat) >= boundary:
            decision = "reject_null"
            crossed_boundary = True
        else:
            decision = "continue"
            crossed_boundary = False
        
        # Adjusted p-value
        adjusted_p = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        # Effect and confidence interval
        effect_estimate = test_stat / sqrt(sample_size / 4)
        ci_width = boundary / sqrt(sample_size / 4)
        
        return SequentialResult(
            analysis_time=analysis_time,
            interim_number=len(self.test_histories.get('alpha_spending', [])) + 1,
            sample_size=sample_size,
            test_statistic=test_stat,
            p_value=adjusted_p,
            lower_bound=-boundary,
            upper_bound=boundary,
            crossed_boundary=crossed_boundary,
            decision=decision,
            confidence_interval=(effect_estimate - ci_width, effect_estimate + ci_width),
            power=self._calculate_conditional_power(test_stat, info_fraction, design)
        )
    
    def _obrien_fleming_boundary(self, t: float, alpha: float) -> float:
        """Calculate O'Brien-Fleming boundary"""
        if t <= 0:
            return float('inf')
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        return z_alpha / sqrt(t)
    
    def _pocock_boundary(self, t: float, alpha: float) -> float:
        """Calculate Pocock boundary (simplified)"""
        # Simplified approximation - actual calculation requires solving boundary equation
        return stats.norm.ppf(1 - alpha / 2) * 1.2  # Conservative approximation
    
    def _obrien_fleming_alpha_spending(self, t: float, alpha: float) -> float:
        """O'Brien-Fleming alpha spending function"""
        if t <= 0:
            return 0.0
        elif t >= 1:
            return alpha
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        return 2 * (1 - stats.norm.cdf(z_alpha / sqrt(t)))
    
    def _calculate_adjusted_p_value(
        self,
        test_stat: float,
        info_fraction: float,
        design: SequentialDesign
    ) -> float:
        """Calculate adjusted p-value for sequential testing"""
        # Simplified adjustment - actual calculation depends on specific boundary
        raw_p = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        # Conservative Bonferroni-like adjustment
        adjustment_factor = 1 / info_fraction
        return min(1.0, raw_p * adjustment_factor)
    
    def _calculate_conditional_power(
        self,
        current_z: float,
        info_fraction: float,
        design: SequentialDesign
    ) -> float:
        """Calculate conditional power to detect effect"""
        if info_fraction >= 1.0:
            return 1.0 if abs(current_z) > stats.norm.ppf(1 - design.alpha / 2) else 0.0
        
        # Remaining information fraction
        remaining_t = 1 - info_fraction
        
        if remaining_t <= 0:
            return 1.0 if abs(current_z) > stats.norm.ppf(1 - design.alpha / 2) else 0.0
        
        # Expected final Z under alternative
        expected_final_z = current_z * sqrt(info_fraction) + design.theta_1 * sqrt(remaining_t)
        
        # Conditional power
        final_boundary = stats.norm.ppf(1 - design.alpha / 2)
        power = 1 - stats.norm.cdf(final_boundary - expected_final_z)
        
        return max(0.0, min(1.0, power))
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of sequential test"""
        
        if test_id not in self.designs:
            raise ValueError(f"Test {test_id} not found")
        
        design = self.designs[test_id]
        history = self.test_histories[test_id]
        test_info = self.active_tests[test_id]
        
        summary = {
            'test_id': test_id,
            'test_type': design.test_type.value,
            'boundary_type': design.boundary_type.value,
            'status': test_info['status'],
            'created_at': test_info['created_at'],
            'total_analyses': len(history),
            'design_parameters': {
                'alpha': design.alpha,
                'beta': design.beta,
                'max_sample_size': design.max_sample_size,
                'effect_size': design.theta_1
            }
        }
        
        if history:
            latest = history[-1]
            summary['current_status'] = {
                'sample_size': latest.sample_size,
                'test_statistic': latest.test_statistic,
                'p_value': latest.p_value,
                'decision': latest.decision,
                'power': latest.power,
                'confidence_interval': latest.confidence_interval
            }
            
            # Analysis timeline
            summary['analysis_timeline'] = [
                {
                    'analysis_number': i + 1,
                    'timestamp': result.analysis_time,
                    'sample_size': result.sample_size,
                    'decision': result.decision,
                    'test_statistic': result.test_statistic
                }
                for i, result in enumerate(history)
            ]
        
        return summary
    
    def calculate_sample_size_trajectory(
        self,
        test_id: str,
        effect_size: float,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Calculate expected sample size trajectory via simulation"""
        
        if test_id not in self.designs:
            raise ValueError(f"Test {test_id} not found")
        
        design = self.designs[test_id]
        sample_sizes = []
        stop_reasons = []
        
        for _ in range(n_simulations):
            # Simulate test trajectory
            current_n = 0
            max_n = design.max_sample_size
            
            for info_frac in design.interim_analyses:
                interim_n = int(max_n * info_frac)
                
                # Simulate test statistic
                true_effect = effect_size
                z_stat = np.random.normal(
                    true_effect * sqrt(interim_n / 4),
                    1
                )
                
                # Check stopping condition
                if design.boundary_type == SequentialBoundary.OBRIEN_FLEMING:
                    boundary = self._obrien_fleming_boundary(info_frac, design.alpha)
                else:
                    boundary = stats.norm.ppf(1 - design.alpha / 2)
                
                if abs(z_stat) >= boundary:
                    sample_sizes.append(interim_n)
                    stop_reasons.append('efficacy')
                    break
                elif z_stat < -boundary * 0.5 and info_frac > 0.5:  # Futility
                    sample_sizes.append(interim_n)
                    stop_reasons.append('futility')
                    break
            else:
                # Reached maximum sample size
                sample_sizes.append(max_n)
                stop_reasons.append('max_size')
        
        return {
            'expected_sample_size': np.mean(sample_sizes),
            'median_sample_size': np.median(sample_sizes),
            'sample_size_quartiles': np.percentile(sample_sizes, [25, 75]),
            'probability_early_stop': np.mean([n < design.max_sample_size for n in sample_sizes]),
            'stop_reason_distribution': {
                reason: stop_reasons.count(reason) / len(stop_reasons)
                for reason in set(stop_reasons)
            },
            'savings_vs_fixed': 1 - (np.mean(sample_sizes) / design.max_sample_size)
        }

