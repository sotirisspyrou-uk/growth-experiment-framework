import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from datetime import datetime, timedelta
import json


class TestStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"


class AllocationMethod(Enum):
    EQUAL = "equal"
    WEIGHTED = "weighted"
    BANDIT = "bandit"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class VariantConfig:
    variant_id: str
    name: str
    description: str
    traffic_allocation: float
    feature_flags: Dict[str, Any]
    implementation_details: Dict[str, Any]


@dataclass
class TestResult:
    variant_id: str
    impressions: int
    conversions: int
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    effect_size: float


@dataclass
class MultiVariantResult:
    test_id: str
    status: TestStatus
    winner: Optional[str]
    results: List[TestResult]
    overall_significance: bool
    test_statistics: Dict[str, float]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]


class MultiVariantTester:
    """Multi-armed bandit and A/B/n testing with dynamic traffic allocation"""
    
    def __init__(self):
        self.active_tests: Dict[str, Dict] = {}
        self.completed_tests: Dict[str, MultiVariantResult] = {}
        
    def create_multivariant_test(
        self,
        test_id: str,
        test_name: str,
        variants: List[VariantConfig],
        target_metrics: List[str],
        allocation_method: AllocationMethod = AllocationMethod.EQUAL,
        min_sample_size: int = 1000,
        significance_level: float = 0.05,
        power: float = 0.8,
        max_duration_days: int = 30
    ) -> str:
        """Create a new multi-variant test"""
        
        # Validate configuration
        self._validate_test_config(variants, allocation_method)
        
        # Calculate required sample sizes
        required_samples = self._calculate_required_sample_sizes(
            variants, min_sample_size, significance_level, power
        )
        
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'variants': {v.variant_id: v for v in variants},
            'target_metrics': target_metrics,
            'allocation_method': allocation_method,
            'min_sample_size': max(min_sample_size, required_samples),
            'significance_level': significance_level,
            'power': power,
            'max_duration_days': max_duration_days,
            'status': TestStatus.PLANNING,
            'created_at': datetime.now(),
            'started_at': None,
            'current_allocations': self._initialize_traffic_allocation(variants, allocation_method),
            'performance_data': {v.variant_id: {'impressions': 0, 'conversions': 0} for v in variants},
            'daily_performance': []
        }
        
        self.active_tests[test_id] = test_config
        return test_id
    
    def start_test(self, test_id: str) -> bool:
        """Start a multi-variant test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test = self.active_tests[test_id]
        test['status'] = TestStatus.ACTIVE
        test['started_at'] = datetime.now()
        
        # Initialize tracking and feature flags
        self._initialize_test_infrastructure(test)
        
        return True
    
    def update_test_data(
        self,
        test_id: str,
        variant_data: Dict[str, Dict[str, int]],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Update test data and return current analysis"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test = self.active_tests[test_id]
        timestamp = timestamp or datetime.now()
        
        # Update performance data
        for variant_id, metrics in variant_data.items():
            if variant_id in test['performance_data']:
                test['performance_data'][variant_id]['impressions'] += metrics.get('impressions', 0)
                test['performance_data'][variant_id]['conversions'] += metrics.get('conversions', 0)
        
        # Store daily snapshot
        daily_snapshot = {
            'date': timestamp,
            'performance': dict(test['performance_data']),
            'allocations': dict(test['current_allocations'])
        }
        test['daily_performance'].append(daily_snapshot)
        
        # Analyze current results
        analysis = self._analyze_test_performance(test_id)
        
        # Update traffic allocation if using adaptive methods
        if test['allocation_method'] in [AllocationMethod.BANDIT, AllocationMethod.PERFORMANCE_BASED]:
            new_allocations = self._optimize_traffic_allocation(test_id, analysis)
            test['current_allocations'] = new_allocations
        
        return analysis
    
    def analyze_test(self, test_id: str) -> MultiVariantResult:
        """Perform comprehensive analysis of test results"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test = self.active_tests[test_id]
        
        # Calculate results for each variant
        variant_results = []
        conversion_rates = []
        sample_sizes = []
        
        control_variant = None
        for variant_id, perf_data in test['performance_data'].items():
            impressions = perf_data['impressions']
            conversions = perf_data['conversions']
            
            if impressions == 0:
                continue
                
            conversion_rate = conversions / impressions
            conversion_rates.append(conversion_rate)
            sample_sizes.append(impressions)
            
            # Calculate confidence interval
            ci = self._calculate_proportion_confidence_interval(
                conversions, impressions, test['significance_level']
            )
            
            variant_result = TestResult(
                variant_id=variant_id,
                impressions=impressions,
                conversions=conversions,
                conversion_rate=conversion_rate,
                confidence_interval=ci,
                statistical_significance=False,  # Will be updated below
                p_value=1.0,  # Will be updated below
                effect_size=0.0  # Will be updated below
            )
            
            variant_results.append(variant_result)
            
            # Identify control variant (typically first or largest allocation)
            if control_variant is None:
                control_variant = variant_result
        
        # Perform statistical tests
        overall_significant, test_stats = self._perform_multivariant_test(
            variant_results, test['significance_level']
        )
        
        # Calculate pairwise comparisons against control
        for result in variant_results:
            if result.variant_id != control_variant.variant_id:
                p_value, effect_size = self._compare_variants(control_variant, result)
                result.p_value = p_value
                result.statistical_significance = p_value < test['significance_level']
                result.effect_size = effect_size
        
        # Determine winner
        winner = self._determine_winner(variant_results, test['significance_level'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            variant_results, winner, test, overall_significant
        )
        
        # Risk assessment
        risk_assessment = self._assess_risks(variant_results, test)
        
        result = MultiVariantResult(
            test_id=test_id,
            status=test['status'],
            winner=winner,
            results=variant_results,
            overall_significance=overall_significant,
            test_statistics=test_stats,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
        
        return result
    
    def should_stop_test_early(self, test_id: str) -> Dict[str, Any]:
        """Check if test should be stopped early for efficacy or futility"""
        analysis = self.analyze_test(test_id)
        test = self.active_tests[test_id]
        
        # Check for early efficacy stopping
        efficacy_stop = self._check_early_efficacy_stopping(analysis, test)
        
        # Check for futility stopping  
        futility_stop = self._check_futility_stopping(analysis, test)
        
        # Check minimum sample size and duration
        min_samples_met = all(
            r.impressions >= test['min_sample_size'] / len(analysis.results)
            for r in analysis.results
        )
        
        days_running = (datetime.now() - test['started_at']).days if test['started_at'] else 0
        min_duration_met = days_running >= 7  # Minimum 1 week
        
        return {
            'stop_for_efficacy': efficacy_stop['should_stop'] and min_samples_met and min_duration_met,
            'stop_for_futility': futility_stop['should_stop'] and min_duration_met,
            'efficacy_details': efficacy_stop,
            'futility_details': futility_stop,
            'min_samples_met': min_samples_met,
            'min_duration_met': min_duration_met,
            'days_running': days_running
        }
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> MultiVariantResult:
        """Stop a test and return final results"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test = self.active_tests[test_id]
        
        # Perform final analysis
        final_result = self.analyze_test(test_id)
        
        # Update status
        if "early" in reason.lower():
            test['status'] = TestStatus.STOPPED_EARLY
            final_result.status = TestStatus.STOPPED_EARLY
        else:
            test['status'] = TestStatus.COMPLETED
            final_result.status = TestStatus.COMPLETED
        
        # Archive test
        test['stopped_at'] = datetime.now()
        test['stop_reason'] = reason
        self.completed_tests[test_id] = final_result
        
        # Clean up test infrastructure
        self._cleanup_test_infrastructure(test)
        
        return final_result
    
    def _validate_test_config(self, variants: List[VariantConfig], allocation_method: AllocationMethod) -> bool:
        """Validate test configuration"""
        if len(variants) < 2:
            raise ValueError("Test must have at least 2 variants")
            
        total_allocation = sum(v.traffic_allocation for v in variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError("Variant traffic allocations must sum to 1.0")
            
        return True
    
    def _calculate_required_sample_sizes(
        self,
        variants: List[VariantConfig],
        min_sample_size: int,
        significance_level: float,
        power: float
    ) -> int:
        """Calculate required sample size for multi-variant test"""
        # Use Bonferroni correction for multiple comparisons
        adjusted_alpha = significance_level / (len(variants) - 1)
        
        # Conservative estimate assuming 5% baseline conversion rate
        baseline_rate = 0.05
        min_detectable_effect = 0.2  # 20% relative improvement
        
        # Calculate sample size using normal approximation
        z_alpha = stats.norm.ppf(1 - adjusted_alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + min_detectable_effect)
        
        pooled_p = (p1 + p2) / 2
        
        sample_size = (
            (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p2 - p1) ** 2
        
        return max(int(np.ceil(sample_size)), min_sample_size)
    
    def _initialize_traffic_allocation(
        self,
        variants: List[VariantConfig],
        allocation_method: AllocationMethod
    ) -> Dict[str, float]:
        """Initialize traffic allocation based on method"""
        if allocation_method == AllocationMethod.EQUAL:
            equal_split = 1.0 / len(variants)
            return {v.variant_id: equal_split for v in variants}
        else:
            return {v.variant_id: v.traffic_allocation for v in variants}
    
    def _initialize_test_infrastructure(self, test: Dict) -> None:
        """Initialize tracking and feature flags for test"""
        # This would integrate with actual feature flag and tracking systems
        pass
    
    def _analyze_test_performance(self, test_id: str) -> Dict[str, Any]:
        """Analyze current test performance"""
        test = self.active_tests[test_id]
        
        analysis = {
            'test_id': test_id,
            'status': test['status'].value,
            'days_running': (datetime.now() - test['started_at']).days if test['started_at'] else 0,
            'variants': {},
            'overall_metrics': {}
        }
        
        total_impressions = sum(p['impressions'] for p in test['performance_data'].values())
        total_conversions = sum(p['conversions'] for p in test['performance_data'].values())
        
        analysis['overall_metrics'] = {
            'total_impressions': total_impressions,
            'total_conversions': total_conversions,
            'overall_conversion_rate': total_conversions / max(total_impressions, 1)
        }
        
        # Analyze each variant
        for variant_id, perf_data in test['performance_data'].items():
            impressions = perf_data['impressions']
            conversions = perf_data['conversions']
            
            variant_analysis = {
                'impressions': impressions,
                'conversions': conversions,
                'conversion_rate': conversions / max(impressions, 1),
                'sample_progress': impressions / test['min_sample_size'],
                'current_allocation': test['current_allocations'][variant_id]
            }
            
            analysis['variants'][variant_id] = variant_analysis
        
        return analysis
    
    def _optimize_traffic_allocation(self, test_id: str, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Optimize traffic allocation based on performance"""
        test = self.active_tests[test_id]
        
        if test['allocation_method'] == AllocationMethod.BANDIT:
            return self._bandit_allocation(analysis)
        elif test['allocation_method'] == AllocationMethod.PERFORMANCE_BASED:
            return self._performance_based_allocation(analysis)
        else:
            return test['current_allocations']
    
    def _bandit_allocation(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Thompson Sampling for bandit allocation"""
        variant_data = analysis['variants']
        allocations = {}
        
        # Sample from Beta distributions
        samples = {}
        for variant_id, data in variant_data.items():
            alpha = data['conversions'] + 1  # Prior alpha = 1
            beta = data['impressions'] - data['conversions'] + 1  # Prior beta = 1
            samples[variant_id] = np.random.beta(alpha, beta)
        
        # Allocate more traffic to higher performing variants
        total_sample = sum(samples.values())
        exploration_rate = 0.1  # 10% exploration
        
        for variant_id, sample in samples.items():
            exploit_allocation = sample / total_sample
            equal_allocation = 1.0 / len(samples)
            
            # Mix exploitation and exploration
            allocations[variant_id] = (
                (1 - exploration_rate) * exploit_allocation +
                exploration_rate * equal_allocation
            )
        
        return allocations
    
    def _performance_based_allocation(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Performance-based allocation with confidence consideration"""
        variant_data = analysis['variants']
        allocations = {}
        
        # Calculate performance scores with confidence adjustment
        scores = {}
        for variant_id, data in variant_data.items():
            if data['impressions'] == 0:
                scores[variant_id] = 0.5  # Neutral score for no data
                continue
            
            conversion_rate = data['conversion_rate']
            
            # Adjust for sample size (Wilson confidence interval)
            n = data['impressions']
            p = conversion_rate
            z = 1.96  # 95% confidence
            
            # Wilson score interval lower bound
            wilson_lower = (
                p + z**2 / (2 * n) - z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
            ) / (1 + z**2 / n)
            
            scores[variant_id] = max(wilson_lower, 0)
        
        # Convert scores to allocations
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal allocation if no clear winner
            equal_split = 1.0 / len(scores)
            return {variant_id: equal_split for variant_id in scores.keys()}
        
        # Weighted allocation with minimum 5% for each variant
        min_allocation = 0.05
        remaining_allocation = 1.0 - len(scores) * min_allocation
        
        for variant_id, score in scores.items():
            proportion = score / total_score
            allocations[variant_id] = min_allocation + proportion * remaining_allocation
        
        return allocations
    
    def _perform_multivariant_test(
        self,
        variant_results: List[TestResult],
        significance_level: float
    ) -> Tuple[bool, Dict[str, float]]:
        """Perform chi-square test for multi-variant comparison"""
        if len(variant_results) < 2:
            return False, {}
        
        # Prepare contingency table
        conversions = [r.conversions for r in variant_results]
        impressions = [r.impressions for r in variant_results]
        non_conversions = [imp - conv for imp, conv in zip(impressions, conversions)]
        
        contingency_table = np.array([conversions, non_conversions])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramer's V)
        n = sum(impressions)
        cramers_v = np.sqrt(chi2_stat / (n * (len(variant_results) - 1)))
        
        test_stats = {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'overall_significant': p_value < significance_level
        }
        
        return p_value < significance_level, test_stats
    
    def _compare_variants(self, control: TestResult, treatment: TestResult) -> Tuple[float, float]:
        """Compare two variants using proportion test"""
        # Two-sample proportion test
        count = np.array([control.conversions, treatment.conversions])
        nobs = np.array([control.impressions, treatment.impressions])
        
        if control.impressions == 0 or treatment.impressions == 0:
            return 1.0, 0.0
        
        # Use chi-square test for proportions
        contingency = np.array([
            [control.conversions, control.impressions - control.conversions],
            [treatment.conversions, treatment.impressions - treatment.conversions]
        ])
        
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
        
        # Calculate effect size (Cohen's h)
        p1 = control.conversion_rate
        p2 = treatment.conversion_rate
        
        if p1 == 0 and p2 == 0:
            effect_size = 0.0
        else:
            # Cohen's h for proportions
            effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        return p_value, effect_size
    
    def _calculate_proportion_confidence_interval(
        self,
        conversions: int,
        impressions: int,
        significance_level: float
    ) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for proportion"""
        if impressions == 0:
            return (0.0, 0.0)
        
        p = conversions / impressions
        z = stats.norm.ppf(1 - significance_level / 2)
        n = impressions
        
        # Wilson confidence interval
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / (1 + z**2 / n)
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)
    
    def _determine_winner(self, variant_results: List[TestResult], significance_level: float) -> Optional[str]:
        """Determine the winning variant"""
        significant_variants = [
            r for r in variant_results 
            if r.statistical_significance and r.effect_size > 0
        ]
        
        if not significant_variants:
            return None
        
        # Return variant with highest conversion rate among significant variants
        winner = max(significant_variants, key=lambda x: x.conversion_rate)
        return winner.variant_id
    
    def _generate_recommendations(
        self,
        variant_results: List[TestResult],
        winner: Optional[str],
        test: Dict,
        overall_significant: bool
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not overall_significant:
            recommendations.append(
                "No statistically significant differences detected. Consider running longer or testing larger changes."
            )
        elif winner:
            winner_result = next(r for r in variant_results if r.variant_id == winner)
            lift = (winner_result.conversion_rate - 
                   min(r.conversion_rate for r in variant_results if r.variant_id != winner)) / \
                   min(r.conversion_rate for r in variant_results if r.variant_id != winner)
            
            recommendations.append(
                f"Implement variant '{winner}' - expected lift: {lift:.1%}"
            )
            recommendations.append(
                f"Winner confidence interval: {winner_result.confidence_interval[0]:.1%} - {winner_result.confidence_interval[1]:.1%}"
            )
        
        # Check for concerning patterns
        for result in variant_results:
            if result.impressions < test['min_sample_size'] / len(variant_results) * 0.8:
                recommendations.append(
                    f"Variant '{result.variant_id}' may need more data for reliable results"
                )
        
        return recommendations
    
    def _assess_risks(self, variant_results: List[TestResult], test: Dict) -> Dict[str, Any]:
        """Assess risks and potential issues"""
        risks = {
            'sample_size_risks': [],
            'statistical_risks': [],
            'business_risks': []
        }
        
        # Sample size risks
        min_recommended = test['min_sample_size'] / len(variant_results)
        for result in variant_results:
            if result.impressions < min_recommended * 0.5:
                risks['sample_size_risks'].append(
                    f"Variant '{result.variant_id}' has very low sample size"
                )
        
        # Statistical risks
        significant_count = sum(1 for r in variant_results if r.statistical_significance)
        if significant_count > 1:
            risks['statistical_risks'].append(
                "Multiple variants showing significance - risk of Type I error"
            )
        
        # Business risks
        conversion_rates = [r.conversion_rate for r in variant_results]
        if max(conversion_rates) - min(conversion_rates) > 0.5:  # 50% difference
            risks['business_risks'].append(
                "Large performance differences detected - verify implementation"
            )
        
        return risks
    
    def _check_early_efficacy_stopping(self, analysis: MultiVariantResult, test: Dict) -> Dict[str, Any]:
        """Check if test should stop early for efficacy"""
        if not analysis.overall_significance:
            return {'should_stop': False, 'reason': 'No significance detected'}
        
        if not analysis.winner:
            return {'should_stop': False, 'reason': 'No clear winner'}
        
        winner_result = next(r for r in analysis.results if r.variant_id == analysis.winner)
        
        # Check if winner has strong effect size and tight confidence interval
        effect_size_threshold = 0.2  # Medium effect size
        ci_width = winner_result.confidence_interval[1] - winner_result.confidence_interval[0]
        
        if winner_result.effect_size > effect_size_threshold and ci_width < 0.1:
            return {
                'should_stop': True,
                'reason': 'Strong effect detected with tight confidence interval',
                'winner': analysis.winner,
                'effect_size': winner_result.effect_size
            }
        
        return {'should_stop': False, 'reason': 'Effect not strong enough for early stopping'}
    
    def _check_futility_stopping(self, analysis: MultiVariantResult, test: Dict) -> Dict[str, Any]:
        """Check if test should stop for futility"""
        # Calculate probability of detecting meaningful difference given current data
        best_result = max(analysis.results, key=lambda x: x.conversion_rate)
        control_result = min(analysis.results, key=lambda x: x.conversion_rate)
        
        current_effect = (best_result.conversion_rate - control_result.conversion_rate) / control_result.conversion_rate
        
        # If current best case scenario is below minimum detectable effect
        min_detectable_effect = 0.05  # 5% minimum meaningful improvement
        
        if current_effect < min_detectable_effect / 2:  # Less than half of minimum
            return {
                'should_stop': True,
                'reason': 'Unlikely to detect meaningful difference',
                'current_max_effect': current_effect
            }
        
        return {'should_stop': False, 'reason': 'Still possible to detect meaningful effect'}
    
    def _cleanup_test_infrastructure(self, test: Dict) -> None:
        """Clean up test infrastructure"""
        # This would clean up feature flags, tracking, etc.
        pass
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive test summary"""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            analysis = self._analyze_test_performance(test_id)
            
            return {
                'test_id': test_id,
                'name': test['test_name'],
                'status': test['status'].value,
                'created_at': test['created_at'],
                'started_at': test.get('started_at'),
                'days_running': analysis['days_running'],
                'variants': len(test['variants']),
                'total_impressions': analysis['overall_metrics']['total_impressions'],
                'overall_conversion_rate': analysis['overall_metrics']['overall_conversion_rate'],
                'current_allocations': test['current_allocations'],
                'allocation_method': test['allocation_method'].value
            }
        elif test_id in self.completed_tests:
            result = self.completed_tests[test_id]
            return {
                'test_id': test_id,
                'status': result.status.value,
                'winner': result.winner,
                'overall_significance': result.overall_significance,
                'total_variants': len(result.results),
                'recommendations': result.recommendations
            }
        else:
            raise ValueError(f"Test {test_id} not found")