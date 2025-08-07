import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.special import beta, gamma
import warnings
from abc import ABC, abstractmethod


class PriorType(Enum):
    BETA_UNIFORM = "beta_uniform"
    BETA_JEFFREYS = "beta_jeffreys"
    BETA_INFORMATIVE = "beta_informative"
    NORMAL = "normal"
    GAMMA = "gamma"


class LossFunction(Enum):
    EXPECTED_LOSS = "expected_loss"
    OPPORTUNITY_COST = "opportunity_cost"
    REGRET = "regret"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class PriorParameters:
    prior_type: PriorType
    parameters: Dict[str, float]
    confidence: float = 0.9


@dataclass
class BayesianResult:
    variant_id: str
    posterior_parameters: Dict[str, float]
    posterior_mean: float
    posterior_variance: float
    credible_interval: Tuple[float, float]
    probability_best: float
    expected_loss: float
    bayes_factor: Optional[float] = None


@dataclass
class DecisionMetrics:
    recommended_action: str
    confidence_level: float
    expected_regret: float
    opportunity_cost: float
    value_remaining: float
    stop_probability: float


class BayesianPrior(ABC):
    """Abstract base class for Bayesian priors"""
    
    @abstractmethod
    def get_posterior_parameters(self, successes: int, trials: int) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def sample_posterior(self, successes: int, trials: int, n_samples: int = 1000) -> np.ndarray:
        pass
    
    @abstractmethod
    def posterior_mean(self, successes: int, trials: int) -> float:
        pass
    
    @abstractmethod
    def posterior_variance(self, successes: int, trials: int) -> float:
        pass
    
    @abstractmethod
    def credible_interval(self, successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
        pass


class BetaPrior(BayesianPrior):
    """Beta prior for binomial data (conversion rates)"""
    
    def __init__(self, alpha: float = 1.0, beta_param: float = 1.0):
        self.alpha = alpha
        self.beta_param = beta_param
        
    def get_posterior_parameters(self, successes: int, trials: int) -> Dict[str, float]:
        posterior_alpha = self.alpha + successes
        posterior_beta = self.beta_param + trials - successes
        return {'alpha': posterior_alpha, 'beta': posterior_beta}
    
    def sample_posterior(self, successes: int, trials: int, n_samples: int = 1000) -> np.ndarray:
        params = self.get_posterior_parameters(successes, trials)
        return np.random.beta(params['alpha'], params['beta'], n_samples)
    
    def posterior_mean(self, successes: int, trials: int) -> float:
        params = self.get_posterior_parameters(successes, trials)
        return params['alpha'] / (params['alpha'] + params['beta'])
    
    def posterior_variance(self, successes: int, trials: int) -> float:
        params = self.get_posterior_parameters(successes, trials)
        alpha, beta_p = params['alpha'], params['beta']
        return (alpha * beta_p) / ((alpha + beta_p)**2 * (alpha + beta_p + 1))
    
    def credible_interval(self, successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
        params = self.get_posterior_parameters(successes, trials)
        lower = stats.beta.ppf(alpha / 2, params['alpha'], params['beta'])
        upper = stats.beta.ppf(1 - alpha / 2, params['alpha'], params['beta'])
        return (lower, upper)


class NormalPrior(BayesianPrior):
    """Normal prior for continuous metrics"""
    
    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        self.mean = mean
        self.variance = variance
        
    def get_posterior_parameters(self, data_sum: float, n_observations: int, 
                                data_variance: float = 1.0) -> Dict[str, float]:
        # Assuming known variance for simplicity
        precision = 1 / self.variance
        data_precision = n_observations / data_variance
        
        posterior_precision = precision + data_precision
        posterior_mean = (precision * self.mean + data_precision * (data_sum / n_observations)) / posterior_precision
        posterior_variance = 1 / posterior_precision
        
        return {'mean': posterior_mean, 'variance': posterior_variance}
    
    def sample_posterior(self, data_sum: float, n_observations: int, 
                        data_variance: float = 1.0, n_samples: int = 1000) -> np.ndarray:
        params = self.get_posterior_parameters(data_sum, n_observations, data_variance)
        return np.random.normal(params['mean'], np.sqrt(params['variance']), n_samples)
    
    def posterior_mean(self, data_sum: float, n_observations: int, data_variance: float = 1.0) -> float:
        params = self.get_posterior_parameters(data_sum, n_observations, data_variance)
        return params['mean']
    
    def posterior_variance(self, data_sum: float, n_observations: int, data_variance: float = 1.0) -> float:
        params = self.get_posterior_parameters(data_sum, n_observations, data_variance)
        return params['variance']
    
    def credible_interval(self, data_sum: float, n_observations: int, 
                         data_variance: float = 1.0, alpha: float = 0.05) -> Tuple[float, float]:
        params = self.get_posterior_parameters(data_sum, n_observations, data_variance)
        std = np.sqrt(params['variance'])
        lower = stats.norm.ppf(alpha / 2, params['mean'], std)
        upper = stats.norm.ppf(1 - alpha / 2, params['mean'], std)
        return (lower, upper)


class BayesianOptimizer:
    """Bayesian optimization for A/B testing with decision theory"""
    
    def __init__(
        self,
        loss_function: LossFunction = LossFunction.EXPECTED_LOSS,
        decision_threshold: float = 0.95,
        min_samples: int = 100,
        max_samples: int = 10000
    ):
        self.loss_function = loss_function
        self.decision_threshold = decision_threshold
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.priors: Dict[str, BayesianPrior] = {}
        self.experiment_history: List[Dict[str, Any]] = []
        
    def set_priors(
        self,
        variant_priors: Dict[str, Tuple[PriorType, Dict[str, float]]]
    ) -> None:
        """Set Bayesian priors for each variant"""
        
        for variant_id, (prior_type, params) in variant_priors.items():
            if prior_type == PriorType.BETA_UNIFORM:
                self.priors[variant_id] = BetaPrior(1.0, 1.0)
            elif prior_type == PriorType.BETA_JEFFREYS:
                self.priors[variant_id] = BetaPrior(0.5, 0.5)
            elif prior_type == PriorType.BETA_INFORMATIVE:
                self.priors[variant_id] = BetaPrior(
                    params.get('alpha', 1.0), 
                    params.get('beta', 1.0)
                )
            elif prior_type == PriorType.NORMAL:
                self.priors[variant_id] = NormalPrior(
                    params.get('mean', 0.0),
                    params.get('variance', 1.0)
                )
            else:
                # Default to uniform Beta
                self.priors[variant_id] = BetaPrior(1.0, 1.0)
    
    def analyze_experiment(
        self,
        experiment_data: Dict[str, Dict[str, Union[int, float]]],
        metric_type: str = "conversion_rate"
    ) -> Dict[str, BayesianResult]:
        """Perform Bayesian analysis of experiment results"""
        
        results = {}
        
        # Calculate posteriors for each variant
        for variant_id, data in experiment_data.items():
            if variant_id not in self.priors:
                # Default uniform prior if none specified
                self.priors[variant_id] = BetaPrior(1.0, 1.0)
            
            prior = self.priors[variant_id]
            
            if metric_type == "conversion_rate":
                successes = data.get('conversions', 0)
                trials = data.get('impressions', 0)
                
                if trials == 0:
                    # Handle zero trials case
                    results[variant_id] = BayesianResult(
                        variant_id=variant_id,
                        posterior_parameters={'alpha': 1.0, 'beta': 1.0},
                        posterior_mean=0.5,
                        posterior_variance=1/12,  # Uniform distribution variance
                        credible_interval=(0.0, 1.0),
                        probability_best=1.0 / len(experiment_data),
                        expected_loss=0.5
                    )
                    continue
                
                posterior_params = prior.get_posterior_parameters(successes, trials)
                posterior_mean = prior.posterior_mean(successes, trials)
                posterior_variance = prior.posterior_variance(successes, trials)
                credible_interval = prior.credible_interval(successes, trials)
                
                results[variant_id] = BayesianResult(
                    variant_id=variant_id,
                    posterior_parameters=posterior_params,
                    posterior_mean=posterior_mean,
                    posterior_variance=posterior_variance,
                    credible_interval=credible_interval,
                    probability_best=0.0,  # Will be calculated below
                    expected_loss=0.0  # Will be calculated below
                )
            
            elif metric_type == "continuous":
                # Handle continuous metrics
                data_sum = data.get('total_value', 0.0)
                n_observations = data.get('count', 0)
                data_variance = data.get('variance', 1.0)
                
                if n_observations == 0:
                    results[variant_id] = BayesianResult(
                        variant_id=variant_id,
                        posterior_parameters={'mean': 0.0, 'variance': 1.0},
                        posterior_mean=0.0,
                        posterior_variance=1.0,
                        credible_interval=(-2.0, 2.0),
                        probability_best=1.0 / len(experiment_data),
                        expected_loss=1.0
                    )
                    continue
                
                if isinstance(prior, NormalPrior):
                    posterior_params = prior.get_posterior_parameters(
                        data_sum, n_observations, data_variance
                    )
                    posterior_mean = prior.posterior_mean(data_sum, n_observations, data_variance)
                    posterior_variance = prior.posterior_variance(data_sum, n_observations, data_variance)
                    credible_interval = prior.credible_interval(data_sum, n_observations, data_variance)
                    
                    results[variant_id] = BayesianResult(
                        variant_id=variant_id,
                        posterior_parameters=posterior_params,
                        posterior_mean=posterior_mean,
                        posterior_variance=posterior_variance,
                        credible_interval=credible_interval,
                        probability_best=0.0,
                        expected_loss=0.0
                    )
        
        # Calculate probability each variant is best
        self._calculate_probabilities_best(results, metric_type)
        
        # Calculate expected losses
        self._calculate_expected_losses(results)
        
        # Calculate Bayes factors
        self._calculate_bayes_factors(results, experiment_data)
        
        return results
    
    def _calculate_probabilities_best(
        self,
        results: Dict[str, BayesianResult],
        metric_type: str,
        n_simulations: int = 10000
    ) -> None:
        """Calculate probability each variant is the best using Monte Carlo"""
        
        # Sample from each posterior
        samples = {}
        for variant_id, result in results.items():
            prior = self.priors[variant_id]
            
            if metric_type == "conversion_rate" and isinstance(prior, BetaPrior):
                params = result.posterior_parameters
                samples[variant_id] = np.random.beta(
                    params['alpha'], params['beta'], n_simulations
                )
            elif metric_type == "continuous" and isinstance(prior, NormalPrior):
                params = result.posterior_parameters
                samples[variant_id] = np.random.normal(
                    params['mean'], np.sqrt(params['variance']), n_simulations
                )
            else:
                # Default sampling
                samples[variant_id] = np.random.uniform(0, 1, n_simulations)
        
        # Count how often each variant is best
        sample_matrix = np.array(list(samples.values()))
        best_indices = np.argmax(sample_matrix, axis=0)
        
        variant_ids = list(samples.keys())
        for i, variant_id in enumerate(variant_ids):
            prob_best = np.sum(best_indices == i) / n_simulations
            results[variant_id].probability_best = prob_best
    
    def _calculate_expected_losses(self, results: Dict[str, BayesianResult]) -> None:
        """Calculate expected loss for each variant"""
        
        # Find the variant with highest posterior mean
        best_variant_mean = max(result.posterior_mean for result in results.values())
        
        for variant_id, result in results.items():
            # Simple expected loss as difference from best
            expected_loss = best_variant_mean - result.posterior_mean
            results[variant_id].expected_loss = max(0, expected_loss)
    
    def _calculate_bayes_factors(
        self,
        results: Dict[str, BayesianResult],
        experiment_data: Dict[str, Dict[str, Union[int, float]]]
    ) -> None:
        """Calculate Bayes factors for model comparison"""
        
        # Simple BF calculation comparing against uniform prior
        for variant_id, result in results.items():
            data = experiment_data[variant_id]
            
            if 'conversions' in data and 'impressions' in data:
                successes = data['conversions']
                trials = data['impressions']
                
                if trials > 0:
                    # Marginal likelihood under current posterior vs uniform prior
                    posterior_params = result.posterior_parameters
                    alpha_post = posterior_params['alpha']
                    beta_post = posterior_params['beta']
                    
                    # Log marginal likelihood (simplified)
                    log_ml = (
                        stats.gammaln(alpha_post + beta_post) -
                        stats.gammaln(alpha_post) -
                        stats.gammaln(beta_post) +
                        stats.gammaln(successes + 1) +
                        stats.gammaln(trials - successes + 1) -
                        stats.gammaln(trials + 2)
                    )
                    
                    # Compare to null hypothesis (uniform)
                    log_ml_null = stats.gammaln(2) - 2 * stats.gammaln(1) - stats.gammaln(trials + 2)
                    
                    bayes_factor = np.exp(log_ml - log_ml_null)
                    results[variant_id].bayes_factor = bayes_factor
    
    def make_decision(
        self,
        results: Dict[str, BayesianResult],
        current_sample_sizes: Dict[str, int]
    ) -> DecisionMetrics:
        """Make optimal decision based on Bayesian analysis"""
        
        # Find variant with highest probability of being best
        best_variant = max(results.items(), key=lambda x: x[1].probability_best)
        best_variant_id, best_result = best_variant
        
        # Calculate decision metrics
        confidence_level = best_result.probability_best
        
        # Expected regret if we stop now and choose best variant
        expected_regret = sum(
            result.probability_best * result.expected_loss
            for result in results.values()
            if result.variant_id != best_variant_id
        )
        
        # Opportunity cost of continuing
        total_samples = sum(current_sample_sizes.values())
        opportunity_cost = self._calculate_opportunity_cost(results, total_samples)
        
        # Value of information remaining
        value_remaining = self._calculate_value_of_information(results, current_sample_sizes)
        
        # Probability we should stop
        stop_probability = self._calculate_stop_probability(
            confidence_level, expected_regret, opportunity_cost, value_remaining
        )
        
        # Decision logic
        if confidence_level >= self.decision_threshold and total_samples >= self.min_samples:
            recommended_action = f"Stop and implement {best_variant_id}"
        elif total_samples >= self.max_samples:
            recommended_action = f"Stop due to sample limit - implement {best_variant_id}"
        elif value_remaining > opportunity_cost:
            recommended_action = "Continue collecting data"
        else:
            recommended_action = f"Stop and implement {best_variant_id}"
        
        return DecisionMetrics(
            recommended_action=recommended_action,
            confidence_level=confidence_level,
            expected_regret=expected_regret,
            opportunity_cost=opportunity_cost,
            value_remaining=value_remaining,
            stop_probability=stop_probability
        )
    
    def _calculate_opportunity_cost(
        self,
        results: Dict[str, BayesianResult],
        total_samples: int
    ) -> float:
        """Calculate opportunity cost of continuing experiment"""
        
        # Simple model: cost increases with sample size
        base_cost = 0.001  # Cost per sample
        total_cost = base_cost * total_samples
        
        # Add uncertainty cost
        uncertainty_cost = sum(
            np.sqrt(result.posterior_variance) for result in results.values()
        ) / len(results)
        
        return total_cost + uncertainty_cost
    
    def _calculate_value_of_information(
        self,
        results: Dict[str, BayesianResult],
        current_sample_sizes: Dict[str, int]
    ) -> float:
        """Calculate expected value of collecting more information"""
        
        # Simplified VOI calculation
        # Value is higher when uncertainty is high and differences are unclear
        
        # Measure uncertainty across variants
        uncertainty = np.mean([
            np.sqrt(result.posterior_variance) for result in results.values()
        ])
        
        # Measure how close the top variants are
        prob_best_values = [result.probability_best for result in results.values()]
        prob_best_values.sort(reverse=True)
        
        if len(prob_best_values) > 1:
            competition = 1 - (prob_best_values[0] - prob_best_values[1])
        else:
            competition = 1.0
        
        # VOI decreases as sample sizes increase
        avg_sample_size = np.mean(list(current_sample_sizes.values()))
        sample_discount = 1.0 / (1.0 + avg_sample_size / 1000)
        
        return uncertainty * competition * sample_discount
    
    def _calculate_stop_probability(
        self,
        confidence_level: float,
        expected_regret: float,
        opportunity_cost: float,
        value_remaining: float
    ) -> float:
        """Calculate probability that we should stop the experiment"""
        
        # Bayesian decision theory approach
        stop_utility = confidence_level - expected_regret - opportunity_cost
        continue_utility = value_remaining - opportunity_cost
        
        # Convert to probability using sigmoid
        utility_difference = stop_utility - continue_utility
        stop_probability = 1 / (1 + np.exp(-5 * utility_difference))
        
        return stop_probability
    
    def calculate_sample_size_allocation(
        self,
        results: Dict[str, BayesianResult],
        total_new_samples: int
    ) -> Dict[str, int]:
        """Optimally allocate new samples across variants"""
        
        # Thompson sampling approach
        allocation = {}
        
        # Sample from each posterior to determine allocation
        samples = []
        variant_ids = list(results.keys())
        
        for _ in range(1000):  # Monte Carlo samples
            variant_samples = {}
            for variant_id, result in results.items():
                # Sample from posterior
                params = result.posterior_parameters
                if 'alpha' in params:  # Beta distribution
                    sample = np.random.beta(params['alpha'], params['beta'])
                else:  # Normal distribution
                    sample = np.random.normal(params['mean'], np.sqrt(params['variance']))
                variant_samples[variant_id] = sample
            
            # Find best variant in this sample
            best_variant = max(variant_samples.items(), key=lambda x: x[1])[0]
            samples.append(best_variant)
        
        # Allocate samples based on probability of being best
        for variant_id in variant_ids:
            prob_best = samples.count(variant_id) / len(samples)
            
            # Mix with equal allocation for exploration
            exploration_rate = 0.2
            equal_prob = 1.0 / len(variant_ids)
            final_prob = (1 - exploration_rate) * prob_best + exploration_rate * equal_prob
            
            allocation[variant_id] = max(1, int(total_new_samples * final_prob))
        
        # Ensure total allocation equals target
        total_allocated = sum(allocation.values())
        if total_allocated != total_new_samples:
            # Adjust largest allocation
            largest_variant = max(allocation.items(), key=lambda x: x[1])[0]
            allocation[largest_variant] += total_new_samples - total_allocated
        
        return allocation
    
    def get_experiment_summary(
        self,
        results: Dict[str, BayesianResult],
        decision_metrics: DecisionMetrics
    ) -> Dict[str, Any]:
        """Generate comprehensive experiment summary"""
        
        # Find best variant
        best_variant = max(results.items(), key=lambda x: x[1].probability_best)
        best_variant_id, best_result = best_variant
        
        # Calculate credible intervals
        credible_intervals = {
            variant_id: result.credible_interval
            for variant_id, result in results.items()
        }
        
        # Posterior means
        posterior_means = {
            variant_id: result.posterior_mean
            for variant_id, result in results.items()
        }
        
        # Probabilities of being best
        probabilities_best = {
            variant_id: result.probability_best
            for variant_id, result in results.items()
        }
        
        return {
            'best_variant': {
                'variant_id': best_variant_id,
                'probability_best': best_result.probability_best,
                'posterior_mean': best_result.posterior_mean,
                'credible_interval': best_result.credible_interval
            },
            'all_variants': {
                'posterior_means': posterior_means,
                'credible_intervals': credible_intervals,
                'probabilities_best': probabilities_best
            },
            'decision': {
                'recommended_action': decision_metrics.recommended_action,
                'confidence_level': decision_metrics.confidence_level,
                'expected_regret': decision_metrics.expected_regret,
                'stop_probability': decision_metrics.stop_probability
            },
            'bayesian_metrics': {
                'total_probability_mass': sum(probabilities_best.values()),
                'uncertainty_measure': np.mean([
                    np.sqrt(result.posterior_variance) for result in results.values()
                ]),
                'bayes_factors': {
                    variant_id: result.bayes_factor
                    for variant_id, result in results.items()
                    if result.bayes_factor is not None
                }
            }
        }
    
    def sequential_update(
        self,
        variant_id: str,
        new_successes: int,
        new_trials: int,
        current_results: Optional[BayesianResult] = None
    ) -> BayesianResult:
        """Update posterior with new data sequentially"""
        
        if variant_id not in self.priors:
            self.priors[variant_id] = BetaPrior(1.0, 1.0)
        
        prior = self.priors[variant_id]
        
        if current_results is None:
            # First update
            total_successes = new_successes
            total_trials = new_trials
        else:
            # Sequential update
            current_params = current_results.posterior_parameters
            total_successes = current_params['alpha'] - 1 + new_successes  # Remove prior
            total_trials = current_params['alpha'] + current_params['beta'] - 2 + new_trials
        
        # Calculate new posterior
        posterior_params = prior.get_posterior_parameters(total_successes, total_trials)
        posterior_mean = prior.posterior_mean(total_successes, total_trials)
        posterior_variance = prior.posterior_variance(total_successes, total_trials)
        credible_interval = prior.credible_interval(total_successes, total_trials)
        
        return BayesianResult(
            variant_id=variant_id,
            posterior_parameters=posterior_params,
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            credible_interval=credible_interval,
            probability_best=0.0,  # Would need all variants to calculate
            expected_loss=0.0  # Would need all variants to calculate
        )
    
    def calculate_lift_probability(
        self,
        control_result: BayesianResult,
        treatment_result: BayesianResult,
        min_lift: float = 0.0
    ) -> float:
        """Calculate probability that treatment lifts control by at least min_lift"""
        
        # Monte Carlo estimation
        n_samples = 10000
        
        # Sample from posteriors
        control_params = control_result.posterior_parameters
        treatment_params = treatment_result.posterior_parameters
        
        control_samples = np.random.beta(
            control_params['alpha'], control_params['beta'], n_samples
        )
        treatment_samples = np.random.beta(
            treatment_params['alpha'], treatment_params['beta'], n_samples
        )
        
        # Calculate lift samples
        lift_samples = (treatment_samples - control_samples) / control_samples
        
        # Probability of lift >= min_lift
        probability = np.mean(lift_samples >= min_lift)
        
        return probability
    
    def expected_lift_distribution(
        self,
        control_result: BayesianResult,
        treatment_result: BayesianResult,
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """Get distribution statistics for expected lift"""
        
        # Sample from posteriors
        control_params = control_result.posterior_parameters
        treatment_params = treatment_result.posterior_parameters
        
        control_samples = np.random.beta(
            control_params['alpha'], control_params['beta'], n_samples
        )
        treatment_samples = np.random.beta(
            treatment_params['alpha'], treatment_params['beta'], n_samples
        )
        
        # Calculate lift samples
        lift_samples = (treatment_samples - control_samples) / np.maximum(control_samples, 1e-10)
        
        return {
            'mean': np.mean(lift_samples),
            'median': np.median(lift_samples),
            'std': np.std(lift_samples),
            'percentile_5': np.percentile(lift_samples, 5),
            'percentile_25': np.percentile(lift_samples, 25),
            'percentile_75': np.percentile(lift_samples, 75),
            'percentile_95': np.percentile(lift_samples, 95),
            'probability_positive': np.mean(lift_samples > 0),
            'probability_negative': np.mean(lift_samples < 0)
        }