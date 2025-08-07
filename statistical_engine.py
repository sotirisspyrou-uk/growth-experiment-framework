import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import ttest_power, zt_ind_solve_power
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    sample_size: int


class StatisticalEngine:
    """Core statistical analysis engine for experimentation framework"""
    
    def __init__(self, default_alpha: float = 0.05, default_power: float = 0.8):
        self.default_alpha = default_alpha
        self.default_power = default_power
    
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        min_effect_size: float,
        significance_level: float = None,
        power: float = None,
        two_sided: bool = True
    ) -> int:
        """Calculate required sample size for experiment"""
        alpha = significance_level or self.default_alpha
        beta = 1 - (power or self.default_power)
        
        # For proportion tests
        if 0 < baseline_rate < 1:
            treatment_rate = baseline_rate * (1 + min_effect_size)
            treatment_rate = min(treatment_rate, 0.99)  # Cap at 99%
            
            # Use statsmodels power analysis
            sample_size = zt_ind_solve_power(
                effect_size=self._proportion_effect_size(baseline_rate, treatment_rate),
                power=power or self.default_power,
                alpha=alpha,
                ratio=1.0,
                alternative='two-sided' if two_sided else 'larger'
            )
            return int(np.ceil(sample_size))
        
        # For continuous metrics - Cohen's d effect size
        cohen_d = min_effect_size / np.sqrt(baseline_rate)  # Assuming baseline_rate is variance
        sample_size = ttest_power(
            effect_size=cohen_d,
            power=power or self.default_power,
            alpha=alpha,
            alternative='two-sided' if two_sided else 'larger'
        )
        return int(np.ceil(sample_size))
    
    def analyze_ab_test(
        self, 
        data: pd.DataFrame,
        variants: List[Dict[str, Any]],
        target_metric: str,
        significance_level: float = None
    ) -> Dict[str, Any]:
        """Comprehensive A/B test analysis"""
        alpha = significance_level or self.default_alpha
        
        # Determine metric type
        metric_type = self._determine_metric_type(data[target_metric])
        
        if metric_type == 'binary':
            return self._analyze_proportion_test(data, variants, target_metric, alpha)
        elif metric_type == 'continuous':
            return self._analyze_continuous_test(data, variants, target_metric, alpha)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    
    def _analyze_proportion_test(
        self, 
        data: pd.DataFrame,
        variants: List[Dict[str, Any]], 
        target_metric: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Analyze binary/proportion metrics"""
        variant_names = [v['name'] for v in variants]
        results = {}
        
        # Calculate conversion rates for each variant
        variant_results = {}
        conversions = []
        totals = []
        
        for variant_name in variant_names:
            variant_data = data[data['variant'] == variant_name]
            conversions_count = variant_data[target_metric].sum()
            total_count = len(variant_data)
            conversion_rate = conversions_count / total_count if total_count > 0 else 0
            
            variant_results[variant_name] = {
                'conversions': conversions_count,
                'total': total_count,
                'conversion_rate': conversion_rate,
                'confidence_interval': proportion_confint(
                    conversions_count, total_count, alpha=alpha, method='wilson'
                )
            }
            
            conversions.append(conversions_count)
            totals.append(total_count)
        
        # Perform statistical test
        if len(variant_names) == 2:
            # Two-sample proportion test
            z_stat, p_value = proportions_ztest(conversions, totals)
            
            # Calculate effect size (relative lift)
            control_rate = variant_results[variant_names[0]]['conversion_rate']
            treatment_rate = variant_results[variant_names[1]]['conversion_rate']
            effect_size = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
            
            winner = variant_names[1] if treatment_rate > control_rate and p_value < alpha else None
            
        else:
            # Chi-square test for multiple variants
            contingency_table = np.array([[conv, total - conv] for conv, total in zip(conversions, totals)])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Find best performing variant
            best_variant_idx = np.argmax([r['conversion_rate'] for r in variant_results.values()])
            winner = variant_names[best_variant_idx] if p_value < alpha else None
            
            # Effect size as relative improvement over worst performer
            rates = [r['conversion_rate'] for r in variant_results.values()]
            effect_size = (max(rates) - min(rates)) / min(rates) if min(rates) > 0 else 0
        
        return {
            'variant_results': variant_results,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'winner': winner,
            'confidence_intervals': {name: result['confidence_interval'] 
                                   for name, result in variant_results.items()},
            'test_type': 'proportion_test'
        }
    
    def _analyze_continuous_test(
        self, 
        data: pd.DataFrame,
        variants: List[Dict[str, Any]], 
        target_metric: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Analyze continuous metrics"""
        variant_names = [v['name'] for v in variants]
        variant_results = {}
        variant_data_arrays = []
        
        # Calculate statistics for each variant
        for variant_name in variant_names:
            variant_data = data[data['variant'] == variant_name][target_metric]
            
            mean_val = variant_data.mean()
            std_val = variant_data.std()
            n = len(variant_data)
            
            # Confidence interval for mean
            sem = std_val / np.sqrt(n)
            ci = stats.t.interval(1 - alpha, n - 1, loc=mean_val, scale=sem)
            
            variant_results[variant_name] = {
                'mean': mean_val,
                'std': std_val,
                'count': n,
                'confidence_interval': ci
            }
            
            variant_data_arrays.append(variant_data.values)
        
        # Perform statistical test
        if len(variant_names) == 2:
            # Two-sample t-test
            control_data, treatment_data = variant_data_arrays
            
            # Check normality assumptions
            if self._check_normality(control_data) and self._check_normality(treatment_data):
                t_stat, p_value = ttest_ind(control_data, treatment_data, equal_var=False)
                test_used = 'welch_ttest'
            else:
                # Use Mann-Whitney U test for non-normal data
                u_stat, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
                test_used = 'mann_whitney_u'
            
            # Calculate Cohen's d effect size
            pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) +
                                (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                               (len(control_data) + len(treatment_data) - 2))
            
            effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
            
            winner = variant_names[1] if np.mean(treatment_data) > np.mean(control_data) and p_value < alpha else None
            
        else:
            # ANOVA for multiple variants
            f_stat, p_value = stats.f_oneway(*variant_data_arrays)
            test_used = 'anova'
            
            # Eta-squared effect size
            ss_between = sum([len(arr) * (np.mean(arr) - np.mean(np.concatenate(variant_data_arrays)))**2 
                            for arr in variant_data_arrays])
            ss_total = sum([(val - np.mean(np.concatenate(variant_data_arrays)))**2 
                           for arr in variant_data_arrays for val in arr])
            effect_size = ss_between / ss_total if ss_total > 0 else 0
            
            # Find best performing variant
            means = [np.mean(arr) for arr in variant_data_arrays]
            best_idx = np.argmax(means)
            winner = variant_names[best_idx] if p_value < alpha else None
        
        return {
            'variant_results': variant_results,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'winner': winner,
            'confidence_intervals': {name: result['confidence_interval'] 
                                   for name, result in variant_results.items()},
            'test_type': test_used
        }
    
    def calculate_statistical_power(
        self,
        observed_effect_size: float,
        sample_size: int,
        significance_level: float = None,
        test_type: str = 'ttest'
    ) -> float:
        """Calculate statistical power of test"""
        alpha = significance_level or self.default_alpha
        
        if test_type == 'ttest':
            power = ttest_power(
                effect_size=observed_effect_size,
                nobs=sample_size,
                alpha=alpha,
                alternative='two-sided'
            )
        else:
            # Placeholder for other test types
            power = 0.8
            
        return power
    
    def perform_sequential_test(
        self,
        data: pd.DataFrame,
        target_metric: str,
        alpha_spending_function: str = 'obrien_fleming'
    ) -> Dict[str, Any]:
        """Perform sequential testing with alpha spending"""
        # Implementation for sequential testing
        # This would include alpha spending functions and early stopping rules
        return {
            'continue_test': True,
            'early_stop_for_efficacy': False,
            'early_stop_for_futility': False,
            'adjusted_p_value': 0.05
        }
    
    def correct_multiple_comparisons(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> List[float]:
        """Apply multiple comparison corrections"""
        if method == 'bonferroni':
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg FDR control
            n = len(p_values)
            sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected = [0] * n
            
            for i, (original_idx, p_val) in enumerate(sorted_pvals):
                corrected[original_idx] = min(p_val * n / (i + 1), 1.0)
            
            return corrected
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def _determine_metric_type(self, metric_data: pd.Series) -> str:
        """Determine if metric is binary, continuous, or count"""
        unique_values = metric_data.nunique()
        
        if unique_values == 2 and set(metric_data.unique()).issubset({0, 1}):
            return 'binary'
        elif metric_data.dtype in ['int64', 'float64'] and unique_values > 10:
            return 'continuous'
        else:
            return 'categorical'
    
    def _proportion_effect_size(self, p1: float, p2: float) -> float:
        """Calculate effect size for proportion test (Cohen's h)"""
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        return abs(h)
    
    def _check_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Check if data follows normal distribution using Shapiro-Wilk test"""
        if len(data) < 3:
            return True  # Assume normal for small samples
        
        if len(data) > 5000:
            # Use Kolmogorov-Smirnov test for large samples
            _, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        else:
            # Use Shapiro-Wilk test for smaller samples
            _, p_value = stats.shapiro(data)
        
        return p_value > alpha
    
    def calculate_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'bootstrap'
    ) -> Tuple[float, float]:
        """Calculate confidence interval using specified method"""
        alpha = 1 - confidence_level
        
        if method == 'bootstrap':
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            
            return (lower, upper)
        
        elif method == 'parametric':
            # Parametric confidence interval assuming normal distribution
            mean = np.mean(data)
            sem = stats.sem(data)
            lower, upper = stats.t.interval(confidence_level, len(data) - 1, loc=mean, scale=sem)
            
            return (lower, upper)
        
        else:
            raise ValueError(f"Unknown CI method: {method}")
    
    def bayesian_ab_test(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        prior_alpha: float = 1,
        prior_beta: float = 1
    ) -> Dict[str, Any]:
        """Bayesian A/B test analysis for conversion rates"""
        # For binary data - Beta-Binomial conjugate prior
        control_conversions = np.sum(control_data)
        control_trials = len(control_data)
        
        treatment_conversions = np.sum(treatment_data)
        treatment_trials = len(treatment_data)
        
        # Posterior parameters
        control_alpha_post = prior_alpha + control_conversions
        control_beta_post = prior_beta + control_trials - control_conversions
        
        treatment_alpha_post = prior_alpha + treatment_conversions
        treatment_beta_post = prior_beta + treatment_trials - treatment_conversions
        
        # Monte Carlo simulation for probability that treatment > control
        n_simulations = 10000
        control_samples = np.random.beta(control_alpha_post, control_beta_post, n_simulations)
        treatment_samples = np.random.beta(treatment_alpha_post, treatment_beta_post, n_simulations)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Credible intervals
        control_ci = (np.percentile(control_samples, 2.5), np.percentile(control_samples, 97.5))
        treatment_ci = (np.percentile(treatment_samples, 2.5), np.percentile(treatment_samples, 97.5))
        
        # Expected lift
        expected_lift = np.mean((treatment_samples - control_samples) / control_samples)
        
        return {
            'prob_treatment_better': prob_treatment_better,
            'control_credible_interval': control_ci,
            'treatment_credible_interval': treatment_ci,
            'expected_lift': expected_lift,
            'control_posterior': (control_alpha_post, control_beta_post),
            'treatment_posterior': (treatment_alpha_post, treatment_beta_post)
        }