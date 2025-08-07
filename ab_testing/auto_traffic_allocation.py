import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import warnings


class AllocationStrategy(Enum):
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "upper_confidence_bound"
    EPSILON_GREEDY = "epsilon_greedy"
    GRADIENT_BANDIT = "gradient_bandit"
    DYNAMIC_THOMPSON = "dynamic_thompson"


@dataclass
class VariantPerformance:
    variant_id: str
    impressions: int
    conversions: int
    conversion_rate: float
    confidence_lower: float
    confidence_upper: float
    posterior_alpha: float
    posterior_beta: float
    regret_estimate: float


@dataclass
class AllocationUpdate:
    variant_id: str
    old_allocation: float
    new_allocation: float
    allocation_change: float
    reason: str
    confidence_score: float


class AutoTrafficAllocator:
    """Automated traffic allocation system using multi-armed bandit algorithms"""
    
    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.THOMPSON_SAMPLING,
        exploration_rate: float = 0.1,
        min_allocation: float = 0.05,
        update_frequency: int = 100
    ):
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        self.min_allocation = min_allocation
        self.update_frequency = update_frequency
        self.allocation_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, VariantPerformance]] = []
        
    def initialize_allocation(
        self,
        variant_ids: List[str],
        initial_strategy: str = "equal",
        prior_beliefs: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """Initialize traffic allocation for variants"""
        
        n_variants = len(variant_ids)
        
        if initial_strategy == "equal":
            base_allocation = 1.0 / n_variants
            allocations = {variant_id: base_allocation for variant_id in variant_ids}
            
        elif initial_strategy == "weighted" and prior_beliefs:
            # Weight by prior belief strength
            total_weight = sum(alpha + beta for alpha, beta in prior_beliefs.values())
            allocations = {}
            for variant_id in variant_ids:
                if variant_id in prior_beliefs:
                    alpha, beta = prior_beliefs[variant_id]
                    weight = (alpha + beta) / total_weight
                    allocations[variant_id] = max(weight, self.min_allocation)
                else:
                    allocations[variant_id] = self.min_allocation
            
            # Normalize to sum to 1
            total = sum(allocations.values())
            allocations = {k: v / total for k, v in allocations.items()}
            
        else:
            # Default to equal allocation
            base_allocation = 1.0 / n_variants
            allocations = {variant_id: base_allocation for variant_id in variant_ids}
        
        # Record initial allocation
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'allocations': allocations.copy(),
            'reason': f'Initial allocation using {initial_strategy} strategy'
        })
        
        return allocations
    
    def update_allocation(
        self,
        current_performance: Dict[str, Dict[str, int]],
        current_allocations: Dict[str, float],
        total_traffic: int
    ) -> Dict[str, AllocationUpdate]:
        """Update traffic allocation based on performance"""
        
        # Calculate performance metrics for each variant
        variant_performances = self._calculate_performance_metrics(current_performance)
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performances': variant_performances
        })
        
        # Calculate new allocations based on strategy
        new_allocations = self._calculate_new_allocations(
            variant_performances, current_allocations, total_traffic
        )
        
        # Generate allocation updates
        updates = {}
        for variant_id in current_allocations.keys():
            old_allocation = current_allocations[variant_id]
            new_allocation = new_allocations[variant_id]
            
            updates[variant_id] = AllocationUpdate(
                variant_id=variant_id,
                old_allocation=old_allocation,
                new_allocation=new_allocation,
                allocation_change=new_allocation - old_allocation,
                reason=self._get_allocation_reason(variant_id, variant_performances),
                confidence_score=self._calculate_confidence_score(
                    variant_performances[variant_id]
                )
            )
        
        # Record allocation update
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'allocations': new_allocations.copy(),
            'updates': updates,
            'strategy': self.strategy.value,
            'total_traffic': total_traffic
        })
        
        return updates
    
    def _calculate_performance_metrics(
        self,
        performance_data: Dict[str, Dict[str, int]]
    ) -> Dict[str, VariantPerformance]:
        """Calculate comprehensive performance metrics for each variant"""
        
        variant_performances = {}
        
        for variant_id, data in performance_data.items():
            impressions = data.get('impressions', 0)
            conversions = data.get('conversions', 0)
            
            if impressions == 0:
                # Handle zero impressions case
                variant_performances[variant_id] = VariantPerformance(
                    variant_id=variant_id,
                    impressions=0,
                    conversions=0,
                    conversion_rate=0.0,
                    confidence_lower=0.0,
                    confidence_upper=1.0,
                    posterior_alpha=1.0,  # Uniform prior
                    posterior_beta=1.0,
                    regret_estimate=0.0
                )
                continue
            
            conversion_rate = conversions / impressions
            
            # Calculate Wilson confidence interval
            z = 1.96  # 95% confidence
            n = impressions
            p = conversion_rate
            
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            
            confidence_lower = max(0, center - margin)
            confidence_upper = min(1, center + margin)
            
            # Beta posterior parameters (assuming Beta(1,1) prior)
            posterior_alpha = 1 + conversions
            posterior_beta = 1 + impressions - conversions
            
            # Estimate regret (simplified)
            regret_estimate = self._estimate_regret(
                conversion_rate, confidence_lower, confidence_upper
            )
            
            variant_performances[variant_id] = VariantPerformance(
                variant_id=variant_id,
                impressions=impressions,
                conversions=conversions,
                conversion_rate=conversion_rate,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                posterior_alpha=posterior_alpha,
                posterior_beta=posterior_beta,
                regret_estimate=regret_estimate
            )
        
        return variant_performances
    
    def _calculate_new_allocations(
        self,
        performances: Dict[str, VariantPerformance],
        current_allocations: Dict[str, float],
        total_traffic: int
    ) -> Dict[str, float]:
        """Calculate new allocations based on strategy"""
        
        if self.strategy == AllocationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_allocation(performances)
        elif self.strategy == AllocationStrategy.UCB:
            return self._ucb_allocation(performances, total_traffic)
        elif self.strategy == AllocationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_allocation(performances)
        elif self.strategy == AllocationStrategy.GRADIENT_BANDIT:
            return self._gradient_bandit_allocation(performances, current_allocations)
        elif self.strategy == AllocationStrategy.DYNAMIC_THOMPSON:
            return self._dynamic_thompson_allocation(performances, total_traffic)
        else:
            # Default to Thompson sampling
            return self._thompson_sampling_allocation(performances)
    
    def _thompson_sampling_allocation(
        self,
        performances: Dict[str, VariantPerformance]
    ) -> Dict[str, float]:
        """Thompson Sampling allocation strategy"""
        
        # Sample from Beta distributions
        samples = {}
        for variant_id, perf in performances.items():
            samples[variant_id] = np.random.beta(
                perf.posterior_alpha, perf.posterior_beta
            )
        
        # Allocate based on samples with exploration
        total_sample = sum(samples.values())
        if total_sample == 0:
            # Equal allocation if all samples are 0
            n_variants = len(performances)
            return {variant_id: 1.0 / n_variants for variant_id in performances.keys()}
        
        allocations = {}
        for variant_id, sample in samples.items():
            # Mix exploitation with exploration
            exploit_allocation = sample / total_sample
            equal_allocation = 1.0 / len(performances)
            
            mixed_allocation = (
                (1 - self.exploration_rate) * exploit_allocation +
                self.exploration_rate * equal_allocation
            )
            
            # Ensure minimum allocation
            allocations[variant_id] = max(mixed_allocation, self.min_allocation)
        
        # Normalize to sum to 1
        total = sum(allocations.values())
        return {k: v / total for k, v in allocations.items()}
    
    def _ucb_allocation(
        self,
        performances: Dict[str, VariantPerformance],
        total_traffic: int
    ) -> Dict[str, float]:
        """Upper Confidence Bound allocation strategy"""
        
        ucb_scores = {}
        
        for variant_id, perf in performances.items():
            if perf.impressions == 0:
                # High UCB for unexplored variants
                ucb_scores[variant_id] = float('inf')
            else:
                # UCB1 formula
                confidence_bonus = np.sqrt(
                    2 * np.log(total_traffic) / perf.impressions
                )
                ucb_scores[variant_id] = perf.conversion_rate + confidence_bonus
        
        # Softmax allocation based on UCB scores
        max_score = max(score for score in ucb_scores.values() if score != float('inf'))
        
        # Handle infinite scores
        inf_variants = [k for k, v in ucb_scores.items() if v == float('inf')]
        if inf_variants:
            # Equal allocation among unexplored variants
            unexplored_allocation = 0.5  # Give 50% to unexplored
            remaining_allocation = 1 - unexplored_allocation
            
            allocations = {}
            for variant_id in ucb_scores.keys():
                if variant_id in inf_variants:
                    allocations[variant_id] = unexplored_allocation / len(inf_variants)
                else:
                    # Softmax for explored variants
                    exp_score = np.exp(ucb_scores[variant_id] - max_score)
                    allocations[variant_id] = 0  # Will be normalized below
            
            # Normalize explored variants
            explored_variants = {k: v for k, v in ucb_scores.items() 
                               if k not in inf_variants}
            if explored_variants:
                exp_scores = {k: np.exp(v - max_score) for k, v in explored_variants.items()}
                total_exp = sum(exp_scores.values())
                
                for variant_id in explored_variants.keys():
                    allocations[variant_id] = (
                        remaining_allocation * exp_scores[variant_id] / total_exp
                    )
        else:
            # Standard softmax allocation
            exp_scores = {k: np.exp(v - max_score) for k, v in ucb_scores.items()}
            total_exp = sum(exp_scores.values())
            
            allocations = {
                k: max(v / total_exp, self.min_allocation)
                for k, v in exp_scores.items()
            }
        
        # Ensure minimum allocations and normalize
        for variant_id in allocations.keys():
            allocations[variant_id] = max(allocations[variant_id], self.min_allocation)
        
        total = sum(allocations.values())
        return {k: v / total for k, v in allocations.items()}
    
    def _epsilon_greedy_allocation(
        self,
        performances: Dict[str, VariantPerformance]
    ) -> Dict[str, float]:
        """Epsilon-greedy allocation strategy"""
        
        # Find best performing variant
        best_variant = max(
            performances.keys(),
            key=lambda x: performances[x].conversion_rate
        )
        
        n_variants = len(performances)
        allocations = {}
        
        for variant_id in performances.keys():
            if variant_id == best_variant:
                # Give (1 - epsilon) to best variant
                allocations[variant_id] = (
                    (1 - self.exploration_rate) + 
                    self.exploration_rate / n_variants
                )
            else:
                # Distribute epsilon equally among other variants
                allocations[variant_id] = self.exploration_rate / n_variants
        
        # Ensure minimum allocations
        for variant_id in allocations.keys():
            allocations[variant_id] = max(allocations[variant_id], self.min_allocation)
        
        # Normalize
        total = sum(allocations.values())
        return {k: v / total for k, v in allocations.items()}
    
    def _gradient_bandit_allocation(
        self,
        performances: Dict[str, VariantPerformance],
        current_allocations: Dict[str, float]
    ) -> Dict[str, float]:
        """Gradient bandit allocation strategy"""
        
        # Calculate rewards and baseline
        rewards = {k: v.conversion_rate for k, v in performances.items()}
        baseline = np.mean(list(rewards.values()))
        
        # Update preferences using gradient ascent
        learning_rate = 0.1
        preferences = {}
        
        for variant_id, current_allocation in current_allocations.items():
            reward = rewards[variant_id]
            
            # Gradient update
            gradient = (reward - baseline) * (1 - current_allocation)
            preferences[variant_id] = current_allocation + learning_rate * gradient
        
        # Softmax to get probabilities
        max_pref = max(preferences.values())
        exp_prefs = {k: np.exp(v - max_pref) for k, v in preferences.items()}
        total_exp = sum(exp_prefs.values())
        
        allocations = {
            k: max(v / total_exp, self.min_allocation)
            for k, v in exp_prefs.items()
        }
        
        # Normalize
        total = sum(allocations.values())
        return {k: v / total for k, v in allocations.items()}
    
    def _dynamic_thompson_allocation(
        self,
        performances: Dict[str, VariantPerformance],
        total_traffic: int
    ) -> Dict[str, float]:
        """Dynamic Thompson Sampling with adaptive exploration"""
        
        # Adjust exploration rate based on total traffic
        # Decrease exploration as we gather more data
        adaptive_exploration = max(
            0.01,  # Minimum exploration
            self.exploration_rate * (1000 / max(total_traffic, 1000))
        )
        
        # Sample from Beta distributions
        samples = {}
        for variant_id, perf in performances.items():
            samples[variant_id] = np.random.beta(
                perf.posterior_alpha, perf.posterior_beta
            )
        
        # Allocate based on samples with adaptive exploration
        total_sample = sum(samples.values())
        if total_sample == 0:
            n_variants = len(performances)
            return {variant_id: 1.0 / n_variants for variant_id in performances.keys()}
        
        allocations = {}
        for variant_id, sample in samples.items():
            exploit_allocation = sample / total_sample
            equal_allocation = 1.0 / len(performances)
            
            mixed_allocation = (
                (1 - adaptive_exploration) * exploit_allocation +
                adaptive_exploration * equal_allocation
            )
            
            allocations[variant_id] = max(mixed_allocation, self.min_allocation)
        
        # Normalize
        total = sum(allocations.values())
        return {k: v / total for k, v in allocations.items()}
    
    def _estimate_regret(
        self,
        conversion_rate: float,
        confidence_lower: float,
        confidence_upper: float
    ) -> float:
        """Estimate regret for allocation decisions"""
        
        # Simple regret estimate based on confidence interval width
        # and distance from optimal (assuming optimal is upper bound)
        regret = confidence_upper - conversion_rate
        return regret
    
    def _get_allocation_reason(
        self,
        variant_id: str,
        performances: Dict[str, VariantPerformance]
    ) -> str:
        """Generate human-readable reason for allocation change"""
        
        perf = performances[variant_id]
        
        if perf.impressions == 0:
            return "Unexplored variant - needs initial traffic"
        elif perf.conversion_rate == max(p.conversion_rate for p in performances.values()):
            return "Best performing variant - increased allocation"
        elif perf.confidence_upper - perf.confidence_lower > 0.1:  # Wide CI
            return "High uncertainty - needs more data"
        else:
            return "Standard allocation adjustment based on performance"
    
    def _calculate_confidence_score(self, performance: VariantPerformance) -> float:
        """Calculate confidence score for allocation decision"""
        
        if performance.impressions < 100:
            return 0.3  # Low confidence with little data
        
        # Higher confidence with more data and tighter confidence intervals
        ci_width = performance.confidence_upper - performance.confidence_lower
        data_confidence = min(performance.impressions / 1000, 1.0)  # Max at 1000 impressions
        precision_confidence = max(0.1, 1 - ci_width)  # Better precision = higher confidence
        
        return (data_confidence + precision_confidence) / 2
    
    def get_allocation_recommendations(
        self,
        current_performance: Dict[str, Dict[str, int]],
        current_allocations: Dict[str, float],
        total_traffic: int
    ) -> Dict[str, Any]:
        """Get recommendations for allocation adjustments"""
        
        performances = self._calculate_performance_metrics(current_performance)
        suggested_allocations = self._calculate_new_allocations(
            performances, current_allocations, total_traffic
        )
        
        # Calculate expected impact of reallocation
        expected_impact = self._calculate_expected_impact(
            performances, current_allocations, suggested_allocations, total_traffic
        )
        
        # Generate detailed recommendations
        recommendations = []
        for variant_id, current_alloc in current_allocations.items():
            suggested_alloc = suggested_allocations[variant_id]
            change = suggested_alloc - current_alloc
            
            if abs(change) > 0.05:  # 5% threshold for recommendation
                impact = expected_impact.get(variant_id, 0)
                
                recommendations.append({
                    'variant_id': variant_id,
                    'current_allocation': current_alloc,
                    'suggested_allocation': suggested_alloc,
                    'change': change,
                    'expected_impact': impact,
                    'reason': self._get_allocation_reason(variant_id, performances),
                    'confidence': self._calculate_confidence_score(performances[variant_id])
                })
        
        return {
            'recommendations': recommendations,
            'total_expected_lift': sum(expected_impact.values()),
            'strategy': self.strategy.value,
            'exploration_rate': self.exploration_rate
        }
    
    def _calculate_expected_impact(
        self,
        performances: Dict[str, VariantPerformance],
        current_allocations: Dict[str, float],
        suggested_allocations: Dict[str, float],
        total_traffic: int
    ) -> Dict[str, float]:
        """Calculate expected impact of allocation changes"""
        
        current_expected = sum(
            alloc * performances[variant_id].conversion_rate
            for variant_id, alloc in current_allocations.items()
        )
        
        suggested_expected = sum(
            alloc * performances[variant_id].conversion_rate
            for variant_id, alloc in suggested_allocations.items()
        )
        
        total_impact = (suggested_expected - current_expected) * total_traffic
        
        # Distribute impact by allocation change
        impact_by_variant = {}
        for variant_id in current_allocations.keys():
            allocation_change = suggested_allocations[variant_id] - current_allocations[variant_id]
            variant_impact = (
                allocation_change * performances[variant_id].conversion_rate * total_traffic
            )
            impact_by_variant[variant_id] = variant_impact
        
        return impact_by_variant
    
    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """Get complete allocation history"""
        return self.allocation_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all variants"""
        
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        latest_performance = self.performance_history[-1]['performances']
        
        summary = {
            'total_variants': len(latest_performance),
            'total_impressions': sum(p.impressions for p in latest_performance.values()),
            'total_conversions': sum(p.conversions for p in latest_performance.values()),
            'overall_conversion_rate': 0,
            'best_variant': None,
            'variant_performance': {}
        }
        
        if summary['total_impressions'] > 0:
            summary['overall_conversion_rate'] = (
                summary['total_conversions'] / summary['total_impressions']
            )
        
        # Find best variant
        best_variant_id = max(
            latest_performance.keys(),
            key=lambda x: latest_performance[x].conversion_rate
        )
        summary['best_variant'] = best_variant_id
        
        # Detailed variant performance
        for variant_id, perf in latest_performance.items():
            summary['variant_performance'][variant_id] = {
                'conversion_rate': perf.conversion_rate,
                'impressions': perf.impressions,
                'conversions': perf.conversions,
                'confidence_interval': (perf.confidence_lower, perf.confidence_upper),
                'is_best': variant_id == best_variant_id
            }
        
        return summary
    
    def should_stop_experiment(
        self,
        performances: Dict[str, VariantPerformance],
        min_impressions: int = 1000,
        confidence_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Determine if experiment should be stopped"""
        
        # Check if minimum sample size is met
        total_impressions = sum(p.impressions for p in performances.values())
        min_sample_met = total_impressions >= min_impressions
        
        if not min_sample_met:
            return {
                'should_stop': False,
                'reason': 'Minimum sample size not reached',
                'confidence': 0.0
            }
        
        # Check if there's a clear winner
        conversion_rates = [p.conversion_rate for p in performances.values()]
        best_rate = max(conversion_rates)
        second_best_rate = sorted(conversion_rates, reverse=True)[1] if len(conversion_rates) > 1 else 0
        
        # Statistical significance check using confidence intervals
        best_variant_id = max(
            performances.keys(),
            key=lambda x: performances[x].conversion_rate
        )
        best_variant = performances[best_variant_id]
        
        # Check if best variant's lower bound > others' upper bounds
        is_significant = all(
            best_variant.confidence_lower > perf.confidence_upper
            for variant_id, perf in performances.items()
            if variant_id != best_variant_id
        )
        
        confidence = best_variant.confidence_lower / max(best_variant.confidence_upper, 0.001)
        
        return {
            'should_stop': is_significant and confidence >= confidence_threshold,
            'reason': 'Clear winner detected' if is_significant else 'No clear winner yet',
            'winner': best_variant_id if is_significant else None,
            'confidence': confidence,
            'best_conversion_rate': best_rate,
            'improvement_over_second': (best_rate - second_best_rate) / max(second_best_rate, 0.001)
        }