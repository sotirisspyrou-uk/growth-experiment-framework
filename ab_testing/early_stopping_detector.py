import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import warnings
from math import log, exp, sqrt


class StoppingBoundary(Enum):
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    ALPHA_SPENDING = "alpha_spending"
    BAYESIAN_THRESHOLD = "bayesian_threshold"
    PRACTICAL_SIGNIFICANCE = "practical_significance"


class StoppingReason(Enum):
    EFFICACY = "efficacy"
    FUTILITY = "futility"
    PRACTICAL_INSIGNIFICANCE = "practical_insignificance"
    SAMPLE_SIZE_REACHED = "sample_size_reached"
    DURATION_EXCEEDED = "duration_exceeded"
    SAFETY = "safety"


@dataclass
class StoppingAnalysis:
    should_stop: bool
    stopping_reason: Optional[StoppingReason]
    confidence_level: float
    effect_size: float
    p_value: float
    test_statistic: float
    boundary_crossed: bool
    futility_probability: float
    expected_effect_size: float
    remaining_power: float
    recommendation: str


@dataclass
class InterimResult:
    analysis_time: datetime
    interim_number: int
    sample_size_fraction: float
    cumulative_sample_size: int
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    boundary_value: float
    crossed_efficacy: bool
    crossed_futility: bool
    alpha_spent: float
    beta_spent: float


class AlphaSpendingFunction:
    """Alpha spending functions for sequential testing"""
    
    @staticmethod
    def obrien_fleming(t: float, alpha: float = 0.05) -> float:
        """O'Brien-Fleming alpha spending function"""
        if t <= 0:
            return 0.0
        elif t >= 1:
            return alpha
        else:
            z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            return 2 * (1 - stats.norm.cdf(z_alpha_2 / sqrt(t)))
    
    @staticmethod
    def pocock(t: float, alpha: float = 0.05) -> float:
        """Pocock alpha spending function"""
        if t <= 0:
            return 0.0
        elif t >= 1:
            return alpha
        else:
            return alpha * log(1 + (exp(1) - 1) * t)
    
    @staticmethod
    def linear(t: float, alpha: float = 0.05) -> float:
        """Linear alpha spending function"""
        if t <= 0:
            return 0.0
        elif t >= 1:
            return alpha
        else:
            return alpha * t
    
    @staticmethod
    def power(t: float, alpha: float = 0.05, gamma: float = 1.5) -> float:
        """Power family alpha spending function"""
        if t <= 0:
            return 0.0
        elif t >= 1:
            return alpha
        else:
            return alpha * (t ** gamma)


class EarlyStoppingDetector:
    """Advanced early stopping detection for A/B tests with multiple boundaries"""
    
    def __init__(
        self,
        boundary_type: StoppingBoundary = StoppingBoundary.OBRIEN_FLEMING,
        alpha: float = 0.05,
        beta: float = 0.2,
        min_effect_size: float = 0.02,
        max_interim_analyses: int = 10,
        min_sample_size: int = 100,
        futility_threshold: float = 0.1
    ):
        self.boundary_type = boundary_type
        self.alpha = alpha
        self.beta = beta
        self.min_effect_size = min_effect_size
        self.max_interim_analyses = max_interim_analyses
        self.min_sample_size = min_sample_size
        self.futility_threshold = futility_threshold
        
        # Spending functions
        self.alpha_spending = AlphaSpendingFunction()
        
        # Track interim analyses
        self.interim_results: List[InterimResult] = []
        self.cumulative_alpha_spent = 0.0
        self.cumulative_beta_spent = 0.0
        
    def analyze_for_early_stopping(
        self,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]],
        planned_sample_size: int,
        current_analysis_time: Optional[datetime] = None
    ) -> StoppingAnalysis:
        """Analyze whether to stop experiment early"""
        
        analysis_time = current_analysis_time or datetime.now()
        
        # Calculate current sample sizes and metrics
        control_n = control_data.get('sample_size', control_data.get('impressions', 0))
        treatment_n = treatment_data.get('sample_size', treatment_data.get('impressions', 0))
        
        total_n = control_n + treatment_n
        sample_size_fraction = total_n / planned_sample_size
        
        # Calculate test statistics
        test_result = self._calculate_test_statistics(control_data, treatment_data)
        
        # Check minimum sample size
        if total_n < self.min_sample_size:
            return StoppingAnalysis(
                should_stop=False,
                stopping_reason=None,
                confidence_level=0.0,
                effect_size=test_result['effect_size'],
                p_value=test_result['p_value'],
                test_statistic=test_result['test_statistic'],
                boundary_crossed=False,
                futility_probability=0.0,
                expected_effect_size=0.0,
                remaining_power=1.0,
                recommendation="Continue - minimum sample size not reached"
            )
        
        # Calculate boundaries
        efficacy_boundary = self._calculate_efficacy_boundary(sample_size_fraction)
        futility_boundary = self._calculate_futility_boundary(
            sample_size_fraction, test_result['test_statistic']
        )
        
        # Check efficacy stopping
        crossed_efficacy = abs(test_result['test_statistic']) >= efficacy_boundary
        
        # Check futility stopping
        futility_analysis = self._analyze_futility(
            test_result, sample_size_fraction, planned_sample_size
        )
        crossed_futility = futility_analysis['should_stop_for_futility']
        
        # Update interim results
        interim_result = InterimResult(
            analysis_time=analysis_time,
            interim_number=len(self.interim_results) + 1,
            sample_size_fraction=sample_size_fraction,
            cumulative_sample_size=total_n,
            test_statistic=test_result['test_statistic'],
            p_value=test_result['p_value'],
            effect_size=test_result['effect_size'],
            confidence_interval=test_result['confidence_interval'],
            boundary_value=efficacy_boundary,
            crossed_efficacy=crossed_efficacy,
            crossed_futility=crossed_futility,
            alpha_spent=self.cumulative_alpha_spent,
            beta_spent=self.cumulative_beta_spent
        )
        
        self.interim_results.append(interim_result)
        
        # Make stopping decision
        stopping_decision = self._make_stopping_decision(
            interim_result, futility_analysis, sample_size_fraction
        )
        
        return stopping_decision
    
    def _calculate_test_statistics(
        self, 
        control_data: Dict[str, Union[int, float]], 
        treatment_data: Dict[str, Union[int, float]]
    ) -> Dict[str, float]:
        """Calculate test statistics for current data"""
        
        # Handle conversion rate data
        if 'conversions' in control_data and 'impressions' in control_data:
            return self._proportion_test_statistics(control_data, treatment_data)
        
        # Handle continuous data
        elif 'mean' in control_data and 'std' in control_data:
            return self._continuous_test_statistics(control_data, treatment_data)
        
        else:
            raise ValueError("Unsupported data format")
    
    def _proportion_test_statistics(
        self,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]]
    ) -> Dict[str, float]:
        """Calculate test statistics for proportion data"""
        
        # Extract data
        n1 = control_data['impressions']
        x1 = control_data['conversions']
        n2 = treatment_data['impressions']
        x2 = treatment_data['conversions']
        
        if n1 == 0 or n2 == 0:
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                'pooled_variance': 0.0
            }
        
        # Calculate proportions
        p1 = x1 / n1
        p2 = x2 / n2
        
        # Pooled proportion for variance calculation
        p_pooled = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Test statistic
        if se == 0:
            z_stat = 0.0
        else:
            z_stat = (p2 - p1) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect size (relative difference)
        effect_size = (p2 - p1) / max(p1, 1e-10)
        
        # Confidence interval for difference in proportions
        se_diff = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        
        diff = p2 - p1
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        return {
            'test_statistic': z_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'pooled_variance': se**2
        }
    
    def _continuous_test_statistics(
        self,
        control_data: Dict[str, Union[int, float]],
        treatment_data: Dict[str, Union[int, float]]
    ) -> Dict[str, float]:
        """Calculate test statistics for continuous data"""
        
        # Extract data
        n1 = control_data['sample_size']
        mean1 = control_data['mean']
        std1 = control_data['std']
        
        n2 = treatment_data['sample_size']
        mean2 = treatment_data['mean']
        std2 = treatment_data['std']
        
        if n1 == 0 or n2 == 0:
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                'pooled_variance': 0.0
            }
        
        # Pooled variance
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        
        # Standard error
        se = sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Test statistic
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean2 - mean1) / se
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Effect size (Cohen's d)
        effect_size = (mean2 - mean1) / sqrt(pooled_var)
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        
        diff = mean2 - mean1
        ci_lower = diff - t_critical * se
        ci_upper = diff + t_critical * se
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'pooled_variance': pooled_var
        }
    
    def _calculate_efficacy_boundary(self, t: float) -> float:
        """Calculate efficacy boundary based on boundary type"""
        
        if self.boundary_type == StoppingBoundary.OBRIEN_FLEMING:
            if t <= 0:
                return float('inf')
            z_alpha_2 = stats.norm.ppf(1 - self.alpha / 2)
            return z_alpha_2 / sqrt(t)
        
        elif self.boundary_type == StoppingBoundary.POCOCK:
            # Approximate Pocock boundary
            if t <= 0:
                return float('inf')
            c = self._pocock_constant()
            return c
        
        elif self.boundary_type == StoppingBoundary.ALPHA_SPENDING:
            return self._alpha_spending_boundary(t)
        
        else:
            # Default to O'Brien-Fleming
            if t <= 0:
                return float('inf')
            z_alpha_2 = stats.norm.ppf(1 - self.alpha / 2)
            return z_alpha_2 / sqrt(t)
    
    def _pocock_constant(self) -> float:
        """Calculate Pocock constant for given alpha and max analyses"""
        # Simplified approximation - in practice would use more precise calculation
        K = self.max_interim_analyses
        if K <= 1:
            return stats.norm.ppf(1 - self.alpha / 2)
        elif K == 2:
            return 2.178
        elif K == 3:
            return 2.289
        elif K == 4:
            return 2.361
        elif K == 5:
            return 2.413
        else:
            # General approximation
            return stats.norm.ppf(1 - self.alpha / 2) + 0.5 * log(K)
    
    def _alpha_spending_boundary(self, t: float) -> float:
        """Calculate boundary using alpha spending approach"""
        
        if t <= 0:
            return float('inf')
        
        # Calculate alpha spent so far
        alpha_spent = self.alpha_spending.obrien_fleming(t, self.alpha)
        
        # Calculate incremental alpha
        incremental_alpha = alpha_spent - self.cumulative_alpha_spent
        
        if incremental_alpha <= 0:
            return float('inf')
        
        # Update cumulative alpha spent
        self.cumulative_alpha_spent = alpha_spent
        
        # Calculate boundary
        z_boundary = stats.norm.ppf(1 - incremental_alpha / 2)
        
        return z_boundary
    
    def _calculate_futility_boundary(self, t: float, current_z: float) -> float:
        """Calculate futility boundary"""
        
        if t >= 1.0:
            return 0.0  # No futility stopping at final analysis
        
        # Beta spending approach
        beta_spent = self.beta * t  # Linear beta spending
        incremental_beta = beta_spent - self.cumulative_beta_spent
        
        if incremental_beta <= 0:
            return float('-inf')
        
        self.cumulative_beta_spent = beta_spent
        
        # Calculate boundary based on conditional power
        z_beta = stats.norm.ppf(incremental_beta)
        
        return z_beta
    
    def _analyze_futility(
        self,
        test_result: Dict[str, float],
        t: float,
        planned_sample_size: int
    ) -> Dict[str, Any]:
        """Analyze futility using conditional power"""
        
        current_z = test_result['test_statistic']
        
        # Calculate conditional power
        conditional_power = self._calculate_conditional_power(
            current_z, t, self.min_effect_size
        )
        
        # Check if conditional power is below threshold
        should_stop_for_futility = conditional_power < self.futility_threshold
        
        # Calculate predicted final p-value
        predicted_final_z = current_z / sqrt(t)  # Approximate
        predicted_p_value = 2 * (1 - stats.norm.cdf(abs(predicted_final_z)))
        
        return {
            'should_stop_for_futility': should_stop_for_futility,
            'conditional_power': conditional_power,
            'predicted_p_value': predicted_p_value,
            'probability_of_success': conditional_power
        }
    
    def _calculate_conditional_power(
        self, 
        current_z: float, 
        t: float, 
        assumed_effect: float
    ) -> float:
        """Calculate conditional power given current results"""
        
        if t >= 1.0:
            return 1.0 if abs(current_z) > stats.norm.ppf(1 - self.alpha / 2) else 0.0
        
        # Information fraction remaining
        remaining_t = 1 - t
        
        if remaining_t <= 0:
            return 1.0 if abs(current_z) > stats.norm.ppf(1 - self.alpha / 2) else 0.0
        
        # Expected final test statistic under assumed effect
        expected_final_z = (
            current_z * sqrt(t) + 
            assumed_effect * sqrt(remaining_t) * stats.norm.ppf(1 - self.beta)
        )
        
        # Probability of exceeding final critical value
        final_critical = stats.norm.ppf(1 - self.alpha / 2)
        conditional_power = 1 - stats.norm.cdf(final_critical - expected_final_z)
        
        return max(0.0, min(1.0, conditional_power))
    
    def _make_stopping_decision(
        self,
        interim_result: InterimResult,
        futility_analysis: Dict[str, Any],
        sample_size_fraction: float
    ) -> StoppingAnalysis:
        """Make final stopping decision"""
        
        # Check for efficacy stopping
        if interim_result.crossed_efficacy:
            return StoppingAnalysis(
                should_stop=True,
                stopping_reason=StoppingReason.EFFICACY,
                confidence_level=1 - interim_result.p_value,
                effect_size=interim_result.effect_size,
                p_value=interim_result.p_value,
                test_statistic=interim_result.test_statistic,
                boundary_crossed=True,
                futility_probability=futility_analysis['probability_of_success'],
                expected_effect_size=interim_result.effect_size,
                remaining_power=futility_analysis['conditional_power'],
                recommendation=f"Stop for efficacy - significant result detected (p={interim_result.p_value:.4f})"
            )
        
        # Check for futility stopping
        if futility_analysis['should_stop_for_futility']:
            return StoppingAnalysis(
                should_stop=True,
                stopping_reason=StoppingReason.FUTILITY,
                confidence_level=interim_result.p_value,
                effect_size=interim_result.effect_size,
                p_value=interim_result.p_value,
                test_statistic=interim_result.test_statistic,
                boundary_crossed=False,
                futility_probability=futility_analysis['conditional_power'],
                expected_effect_size=interim_result.effect_size,
                remaining_power=futility_analysis['conditional_power'],
                recommendation=f"Stop for futility - low probability of success ({futility_analysis['conditional_power']:.1%})"
            )
        
        # Check for practical insignificance
        if abs(interim_result.effect_size) < self.min_effect_size and sample_size_fraction > 0.5:
            return StoppingAnalysis(
                should_stop=True,
                stopping_reason=StoppingReason.PRACTICAL_INSIGNIFICANCE,
                confidence_level=interim_result.p_value,
                effect_size=interim_result.effect_size,
                p_value=interim_result.p_value,
                test_statistic=interim_result.test_statistic,
                boundary_crossed=False,
                futility_probability=futility_analysis['conditional_power'],
                expected_effect_size=interim_result.effect_size,
                remaining_power=futility_analysis['conditional_power'],
                recommendation=f"Stop for practical insignificance - effect size ({interim_result.effect_size:.3f}) below threshold ({self.min_effect_size:.3f})"
            )
        
        # Continue testing
        return StoppingAnalysis(
            should_stop=False,
            stopping_reason=None,
            confidence_level=interim_result.p_value,
            effect_size=interim_result.effect_size,
            p_value=interim_result.p_value,
            test_statistic=interim_result.test_statistic,
            boundary_crossed=False,
            futility_probability=futility_analysis['conditional_power'],
            expected_effect_size=interim_result.effect_size,
            remaining_power=futility_analysis['conditional_power'],
            recommendation=f"Continue testing - conditional power: {futility_analysis['conditional_power']:.1%}"
        )
    
    def calculate_optimal_interim_times(
        self,
        planned_sample_size: int,
        max_analyses: int = None
    ) -> List[float]:
        """Calculate optimal interim analysis times"""
        
        max_analyses = max_analyses or self.max_interim_analyses
        
        if self.boundary_type == StoppingBoundary.OBRIEN_FLEMING:
            # O'Brien-Fleming optimal spacing
            # More analyses early when boundaries are high
            times = []
            for i in range(1, max_analyses + 1):
                t = (i / max_analyses) ** 2
                times.append(t)
            return times
        
        elif self.boundary_type == StoppingBoundary.POCOCK:
            # Equal spacing for Pocock
            return [i / max_analyses for i in range(1, max_analyses + 1)]
        
        else:
            # Default equal spacing
            return [i / max_analyses for i in range(1, max_analyses + 1)]
    
    def estimate_expected_sample_size(
        self,
        planned_sample_size: int,
        true_effect_size: float,
        interim_times: List[float] = None
    ) -> Dict[str, float]:
        """Estimate expected sample size under different scenarios"""
        
        interim_times = interim_times or self.calculate_optimal_interim_times(planned_sample_size)
        
        # Monte Carlo simulation to estimate expected sample size
        n_simulations = 1000
        sample_sizes = []
        
        for _ in range(n_simulations):
            # Simulate experiment
            for i, t in enumerate(interim_times):
                current_n = int(planned_sample_size * t)
                
                # Simulate test statistic
                # Simplified - assumes normal distribution
                mean_z = true_effect_size * sqrt(current_n / 4)  # Rough approximation
                simulated_z = np.random.normal(mean_z, 1)
                
                # Check boundaries
                efficacy_boundary = self._calculate_efficacy_boundary(t)
                
                if abs(simulated_z) >= efficacy_boundary:
                    sample_sizes.append(current_n)
                    break
                
                # Check futility (simplified)
                conditional_power = self._calculate_conditional_power(simulated_z, t, true_effect_size)
                if conditional_power < self.futility_threshold:
                    sample_sizes.append(current_n)
                    break
            else:
                # Reached end without stopping
                sample_sizes.append(planned_sample_size)
        
        return {
            'expected_sample_size': np.mean(sample_sizes),
            'median_sample_size': np.median(sample_sizes),
            'min_sample_size': np.min(sample_sizes),
            'max_sample_size': np.max(sample_sizes),
            'probability_early_stop': np.mean([n < planned_sample_size for n in sample_sizes])
        }
    
    def get_boundary_visualization_data(
        self,
        planned_sample_size: int,
        interim_times: List[float] = None
    ) -> Dict[str, List[float]]:
        """Get data for visualizing stopping boundaries"""
        
        interim_times = interim_times or self.calculate_optimal_interim_times(planned_sample_size)
        
        efficacy_boundaries = []
        futility_boundaries = []
        
        for t in interim_times:
            efficacy_boundaries.append(self._calculate_efficacy_boundary(t))
            futility_boundaries.append(self._calculate_futility_boundary(t, 0))
        
        return {
            'interim_times': interim_times,
            'sample_sizes': [int(planned_sample_size * t) for t in interim_times],
            'efficacy_boundaries': efficacy_boundaries,
            'futility_boundaries': futility_boundaries
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all interim analyses"""
        
        if not self.interim_results:
            return {'message': 'No interim analyses performed yet'}
        
        summary = {
            'total_analyses': len(self.interim_results),
            'current_sample_size': self.interim_results[-1].cumulative_sample_size,
            'alpha_spent': self.cumulative_alpha_spent,
            'beta_spent': self.cumulative_beta_spent,
            'analyses': []
        }
        
        for result in self.interim_results:
            summary['analyses'].append({
                'analysis_number': result.interim_number,
                'sample_size': result.cumulative_sample_size,
                'sample_fraction': result.sample_size_fraction,
                'test_statistic': result.test_statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'efficacy_boundary': result.boundary_value,
                'crossed_efficacy': result.crossed_efficacy,
                'crossed_futility': result.crossed_futility
            })
        
        return summary
    
    def reset_analysis(self) -> None:
        """Reset all interim analysis data"""
        self.interim_results.clear()
        self.cumulative_alpha_spent = 0.0
        self.cumulative_beta_spent = 0.0
    
    def validate_stopping_procedure(
        self,
        null_effect_size: float = 0.0,
        alternative_effect_size: float = None,
        n_simulations: int = 1000
    ) -> Dict[str, float]:
        """Validate Type I and Type II error rates via simulation"""
        
        alternative_effect_size = alternative_effect_size or self.min_effect_size
        
        # Simulate under null hypothesis
        null_rejections = 0
        null_sample_sizes = []
        
        for _ in range(n_simulations):
            self.reset_analysis()
            
            interim_times = self.calculate_optimal_interim_times(1000)  # Fixed for simulation
            
            for t in interim_times:
                # Simulate data under null
                z_stat = np.random.normal(0, 1)  # Null hypothesis
                
                # Create fake data for analysis
                fake_control = {'conversions': 50, 'impressions': 1000}
                fake_treatment = {'conversions': 50, 'impressions': 1000}
                
                analysis = self.analyze_for_early_stopping(
                    fake_control, fake_treatment, 2000
                )
                
                if analysis.should_stop:
                    if analysis.stopping_reason == StoppingReason.EFFICACY:
                        null_rejections += 1
                    null_sample_sizes.append(int(2000 * t))
                    break
            else:
                null_sample_sizes.append(2000)
        
        # Simulate under alternative hypothesis
        alt_rejections = 0
        alt_sample_sizes = []
        
        for _ in range(n_simulations):
            self.reset_analysis()
            
            interim_times = self.calculate_optimal_interim_times(1000)
            
            for t in interim_times:
                # Simulate data under alternative
                z_stat = np.random.normal(alternative_effect_size * sqrt(1000 * t / 4), 1)
                
                # Create fake data
                base_rate = 0.05
                improvement = alternative_effect_size
                control_rate = base_rate
                treatment_rate = base_rate * (1 + improvement)
                
                n_per_group = int(1000 * t)
                control_conv = np.random.binomial(n_per_group, control_rate)
                treatment_conv = np.random.binomial(n_per_group, treatment_rate)
                
                fake_control = {'conversions': control_conv, 'impressions': n_per_group}
                fake_treatment = {'conversions': treatment_conv, 'impressions': n_per_group}
                
                analysis = self.analyze_for_early_stopping(
                    fake_control, fake_treatment, 2000
                )
                
                if analysis.should_stop:
                    if analysis.stopping_reason == StoppingReason.EFFICACY:
                        alt_rejections += 1
                    alt_sample_sizes.append(int(2000 * t))
                    break
            else:
                alt_sample_sizes.append(2000)
        
        return {
            'type_i_error': null_rejections / n_simulations,
            'power': alt_rejections / n_simulations,
            'expected_n_null': np.mean(null_sample_sizes),
            'expected_n_alt': np.mean(alt_sample_sizes),
            'early_stop_prob_null': np.mean([n < 2000 for n in null_sample_sizes]),
            'early_stop_prob_alt': np.mean([n < 2000 for n in alt_sample_sizes])
        }