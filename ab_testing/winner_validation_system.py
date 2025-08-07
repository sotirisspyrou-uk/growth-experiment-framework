import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.special import beta
import warnings
from math import sqrt, log


class ValidationMethod(Enum):
    FREQUENTIST = "frequentist"
    BAYESIAN = "bayesian"
    HYBRID = "hybrid"
    BOOTSTRAP = "bootstrap"


class ValidationCriteria(Enum):
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    PRACTICAL_SIGNIFICANCE = "practical_significance"
    BUSINESS_SIGNIFICANCE = "business_significance"
    COMBINED = "combined"


@dataclass
class ValidationConfig:
    method: ValidationMethod
    criteria: ValidationCriteria
    significance_level: float
    min_effect_size: float
    min_sample_size: int
    confidence_level: float
    power_threshold: float
    validation_period_days: int


@dataclass  
class ValidationResult:
    is_valid_winner: bool
    confidence_score: float
    validation_method: str
    statistical_evidence: Dict[str, float]
    business_metrics: Dict[str, float]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    validation_timestamp: datetime


@dataclass
class WinnerMetrics:
    variant_id: str
    sample_size: int
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    business_value: float
    risk_score: float


class WinnerValidationSystem:
    """Comprehensive winner validation with multiple statistical approaches"""
    
    def __init__(self):
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.winner_registry: Dict[str, WinnerMetrics] = {}
        self.validation_configs: Dict[str, ValidationConfig] = {}
        
    def configure_validation(
        self,
        test_id: str,
        method: ValidationMethod = ValidationMethod.HYBRID,
        criteria: ValidationCriteria = ValidationCriteria.COMBINED,
        significance_level: float = 0.05,
        min_effect_size: float = 0.02,
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
        power_threshold: float = 0.8,
        validation_period_days: int = 7
    ) -> str:
        """Configure validation parameters for a test"""
        
        config = ValidationConfig(
            method=method,
            criteria=criteria,
            significance_level=significance_level,
            min_effect_size=min_effect_size,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            power_threshold=power_threshold,
            validation_period_days=validation_period_days
        )
        
        self.validation_configs[test_id] = config
        self.validation_history[test_id] = []
        
        return test_id
    
    def validate_winner(
        self,
        test_id: str,
        winner_data: Dict[str, Union[int, float]],
        control_data: Dict[str, Union[int, float]],
        business_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Comprehensive winner validation"""
        
        if test_id not in self.validation_configs:
            raise ValueError(f"Validation config for test {test_id} not found")
        
        config = self.validation_configs[test_id]
        business_context = business_context or {}
        
        # Extract metrics
        winner_metrics = self._extract_winner_metrics(winner_data, control_data)
        
        # Perform validation based on method
        if config.method == ValidationMethod.FREQUENTIST:
            validation_result = self._frequentist_validation(
                winner_metrics, control_data, config
            )
        elif config.method == ValidationMethod.BAYESIAN:
            validation_result = self._bayesian_validation(
                winner_metrics, control_data, config
            )
        elif config.method == ValidationMethod.BOOTSTRAP:
            validation_result = self._bootstrap_validation(
                winner_metrics, control_data, config
            )
        else:  # HYBRID
            validation_result = self._hybrid_validation(
                winner_metrics, control_data, config
            )
        
        # Apply validation criteria
        final_result = self._apply_validation_criteria(
            validation_result, winner_metrics, config, business_context
        )
        
        # Store validation result
        self.validation_history[test_id].append(final_result)
        
        # Register winner if validated
        if final_result.is_valid_winner:
            self.winner_registry[test_id] = winner_metrics
        
        return final_result
    
    def _extract_winner_metrics(
        self,
        winner_data: Dict[str, Union[int, float]],
        control_data: Dict[str, Union[int, float]]
    ) -> WinnerMetrics:
        """Extract comprehensive metrics for winner"""
        
        # Basic metrics
        variant_id = winner_data.get('variant_id', 'treatment')
        winner_impressions = winner_data.get('impressions', 0)
        winner_conversions = winner_data.get('conversions', 0)
        
        control_impressions = control_data.get('impressions', 0)
        control_conversions = control_data.get('conversions', 0)
        
        if winner_impressions == 0 or control_impressions == 0:
            return WinnerMetrics(
                variant_id=variant_id,
                sample_size=0,
                conversion_rate=0.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                statistical_power=0.0,
                business_value=0.0,
                risk_score=1.0
            )
        
        # Conversion rates
        winner_rate = winner_conversions / winner_impressions
        control_rate = control_conversions / control_impressions
        
        # Effect size (Cohen's h for proportions)
        if control_rate == 0 and winner_rate == 0:
            effect_size = 0.0
        else:
            effect_size = 2 * (np.arcsin(np.sqrt(winner_rate)) - np.arcsin(np.sqrt(control_rate)))
        
        # Confidence interval (Wilson method)
        ci = self._calculate_wilson_confidence_interval(
            winner_conversions, winner_impressions, 0.05
        )
        
        # Statistical power calculation
        power = self._calculate_statistical_power(
            control_rate, winner_rate, control_impressions + winner_impressions
        )
        
        # Business value (simplified)
        business_value = (winner_rate - control_rate) * winner_impressions
        
        # Risk score (higher = more risky)
        risk_score = self._calculate_risk_score(winner_data, control_data)
        
        return WinnerMetrics(
            variant_id=variant_id,
            sample_size=winner_impressions,
            conversion_rate=winner_rate,
            confidence_interval=ci,
            effect_size=effect_size,
            statistical_power=power,
            business_value=business_value,
            risk_score=risk_score
        )
    
    def _frequentist_validation(
        self,
        winner_metrics: WinnerMetrics,
        control_data: Dict[str, Union[int, float]],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Frequentist statistical validation"""
        
        # Two-proportion z-test
        winner_conv = winner_metrics.sample_size * winner_metrics.conversion_rate
        winner_n = winner_metrics.sample_size
        control_conv = control_data.get('conversions', 0)
        control_n = control_data.get('impressions', 0)
        
        if control_n == 0 or winner_n == 0:
            return {
                'is_significant': False,
                'p_value': 1.0,
                'confidence': 0.0,
                'method': 'frequentist'
            }
        
        # Pooled proportion test
        total_conv = winner_conv + control_conv
        total_n = winner_n + control_n
        p_pooled = total_conv / total_n
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/winner_n + 1/control_n))
        
        # Z-statistic
        if se == 0:
            z_stat = 0.0
        else:
            z_stat = (winner_metrics.conversion_rate - control_conv/control_n) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Statistical significance
        is_significant = p_value < config.significance_level
        
        # Confidence level
        confidence = 1 - p_value if is_significant else p_value
        
        return {
            'is_significant': is_significant,
            'p_value': p_value,
            'z_statistic': z_stat,
            'confidence': confidence,
            'method': 'frequentist'
        }
    
    def _bayesian_validation(
        self,
        winner_metrics: WinnerMetrics,
        control_data: Dict[str, Union[int, float]],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Bayesian validation using Beta-Binomial model"""
        
        # Prior parameters (Jeffrey's prior)
        alpha_prior = 0.5
        beta_prior = 0.5
        
        # Winner posterior
        winner_conv = winner_metrics.sample_size * winner_metrics.conversion_rate
        winner_alpha = alpha_prior + winner_conv
        winner_beta = beta_prior + winner_metrics.sample_size - winner_conv
        
        # Control posterior  
        control_conv = control_data.get('conversions', 0)
        control_n = control_data.get('impressions', 0)
        control_alpha = alpha_prior + control_conv
        control_beta = beta_prior + control_n - control_conv
        
        # Monte Carlo sampling for probability winner > control
        n_samples = 10000
        winner_samples = np.random.beta(winner_alpha, winner_beta, n_samples)
        control_samples = np.random.beta(control_alpha, control_beta, n_samples)
        
        prob_winner_better = np.mean(winner_samples > control_samples)
        
        # Credible interval for difference
        diff_samples = winner_samples - control_samples
        credible_interval = np.percentile(diff_samples, [2.5, 97.5])
        
        # Bayes Factor approximation
        marginal_likelihood_alt = beta(winner_alpha, winner_beta) / beta(alpha_prior, beta_prior)
        marginal_likelihood_null = beta(control_alpha, control_beta) / beta(alpha_prior, beta_prior)
        
        if marginal_likelihood_null > 0:
            bayes_factor = marginal_likelihood_alt / marginal_likelihood_null
        else:
            bayes_factor = float('inf')
        
        # Decision threshold
        prob_threshold = 1 - config.significance_level
        is_significant = prob_winner_better > prob_threshold
        
        return {
            'is_significant': is_significant,
            'probability_better': prob_winner_better,
            'credible_interval': credible_interval,
            'bayes_factor': bayes_factor,
            'confidence': prob_winner_better,
            'method': 'bayesian'
        }
    
    def _bootstrap_validation(
        self,
        winner_metrics: WinnerMetrics,
        control_data: Dict[str, Union[int, float]],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Bootstrap validation for non-parametric testing"""
        
        # Simulate original data
        winner_conv = int(winner_metrics.sample_size * winner_metrics.conversion_rate)
        winner_data_sim = [1] * winner_conv + [0] * (winner_metrics.sample_size - winner_conv)
        
        control_conv = control_data.get('conversions', 0)
        control_n = control_data.get('impressions', 0)
        control_data_sim = [1] * control_conv + [0] * (control_n - control_conv)
        
        # Bootstrap resampling
        n_bootstrap = 10000
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            winner_resample = np.random.choice(winner_data_sim, size=len(winner_data_sim), replace=True)
            control_resample = np.random.choice(control_data_sim, size=len(control_data_sim), replace=True)
            
            # Calculate difference in means
            winner_rate_boot = np.mean(winner_resample)
            control_rate_boot = np.mean(control_resample)
            bootstrap_diffs.append(winner_rate_boot - control_rate_boot)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        # Null hypothesis: difference = 0
        p_value = np.mean(np.abs(bootstrap_diffs) >= abs(winner_metrics.conversion_rate - control_conv/control_n)) * 2
        
        # Bootstrap confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Significance test
        is_significant = p_value < config.significance_level
        
        return {
            'is_significant': is_significant,
            'p_value': p_value,
            'bootstrap_ci': (ci_lower, ci_upper),
            'bootstrap_mean': np.mean(bootstrap_diffs),
            'bootstrap_std': np.std(bootstrap_diffs),
            'confidence': 1 - p_value if is_significant else p_value,
            'method': 'bootstrap'
        }
    
    def _hybrid_validation(
        self,
        winner_metrics: WinnerMetrics,
        control_data: Dict[str, Union[int, float]],
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Hybrid validation combining multiple methods"""
        
        # Run all validation methods
        freq_result = self._frequentist_validation(winner_metrics, control_data, config)
        bayes_result = self._bayesian_validation(winner_metrics, control_data, config)
        bootstrap_result = self._bootstrap_validation(winner_metrics, control_data, config)
        
        # Combine results using weighted approach
        methods_agree = (
            freq_result['is_significant'] +
            bayes_result['is_significant'] + 
            bootstrap_result['is_significant']
        )
        
        # Require at least 2/3 methods to agree
        is_significant = methods_agree >= 2
        
        # Combined confidence (average of method confidences)
        combined_confidence = (
            freq_result['confidence'] +
            bayes_result['confidence'] +
            bootstrap_result['confidence']
        ) / 3
        
        # Meta p-value using Fisher's method
        p_values = [
            freq_result['p_value'],
            bootstrap_result['p_value']
        ]
        
        # Fisher's combined p-value
        fisher_stat = -2 * sum(log(p) for p in p_values if p > 0)
        combined_p_value = 1 - stats.chi2.cdf(fisher_stat, 2 * len(p_values))
        
        return {
            'is_significant': is_significant,
            'methods_agreement': methods_agree,
            'combined_confidence': combined_confidence,
            'combined_p_value': combined_p_value,
            'frequentist_result': freq_result,
            'bayesian_result': bayes_result,
            'bootstrap_result': bootstrap_result,
            'confidence': combined_confidence,
            'method': 'hybrid'
        }
    
    def _apply_validation_criteria(
        self,
        validation_result: Dict[str, Any],
        winner_metrics: WinnerMetrics,
        config: ValidationConfig,
        business_context: Dict[str, Any]
    ) -> ValidationResult:
        """Apply validation criteria and generate final result"""
        
        # Statistical significance check
        statistical_valid = validation_result['is_significant']
        
        # Practical significance check
        practical_valid = abs(winner_metrics.effect_size) >= config.min_effect_size
        
        # Sample size check
        sample_size_valid = winner_metrics.sample_size >= config.min_sample_size
        
        # Power check
        power_valid = winner_metrics.statistical_power >= config.power_threshold
        
        # Business significance check (if context provided)
        business_valid = True
        if business_context:
            min_business_value = business_context.get('min_value_threshold', 0)
            business_valid = winner_metrics.business_value >= min_business_value
        
        # Apply criteria
        if config.criteria == ValidationCriteria.STATISTICAL_SIGNIFICANCE:
            is_valid = statistical_valid and sample_size_valid
        elif config.criteria == ValidationCriteria.PRACTICAL_SIGNIFICANCE:
            is_valid = practical_valid and sample_size_valid
        elif config.criteria == ValidationCriteria.BUSINESS_SIGNIFICANCE:
            is_valid = business_valid and sample_size_valid
        else:  # COMBINED
            is_valid = all([
                statistical_valid,
                practical_valid,
                sample_size_valid,
                power_valid,
                business_valid
            ])
        
        # Risk assessment
        risk_assessment = self._assess_implementation_risks(
            winner_metrics, validation_result, business_context
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_valid, winner_metrics, validation_result, config, risk_assessment
        )
        
        # Calculate overall confidence score
        confidence_factors = [
            validation_result['confidence'],
            min(winner_metrics.statistical_power, 1.0),
            max(0, 1 - winner_metrics.risk_score)
        ]
        
        if business_context:
            business_confidence = min(winner_metrics.business_value / 
                                   max(business_context.get('target_value', 1), 1), 1.0)
            confidence_factors.append(business_confidence)
        
        confidence_score = np.mean(confidence_factors)
        
        return ValidationResult(
            is_valid_winner=is_valid,
            confidence_score=confidence_score,
            validation_method=validation_result['method'],
            statistical_evidence={
                'p_value': validation_result.get('combined_p_value', validation_result.get('p_value')),
                'effect_size': winner_metrics.effect_size,
                'confidence_interval': winner_metrics.confidence_interval,
                'statistical_power': winner_metrics.statistical_power
            },
            business_metrics={
                'conversion_rate': winner_metrics.conversion_rate,
                'business_value': winner_metrics.business_value,
                'sample_size': winner_metrics.sample_size
            },
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            validation_timestamp=datetime.now()
        )
    
    def _calculate_wilson_confidence_interval(
        self,
        successes: int,
        trials: int,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for proportion"""
        
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - alpha / 2)
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def _calculate_statistical_power(
        self,
        control_rate: float,
        treatment_rate: float,
        total_sample_size: int
    ) -> float:
        """Calculate statistical power for proportion test"""
        
        if control_rate == 0 or total_sample_size == 0:
            return 0.0
        
        # Effect size
        effect_size = treatment_rate - control_rate
        
        # Pooled standard error under alternative hypothesis
        p_avg = (control_rate + treatment_rate) / 2
        se = sqrt(2 * p_avg * (1 - p_avg) / (total_sample_size / 2))
        
        if se == 0:
            return 1.0 if effect_size > 0 else 0.0
        
        # Critical value for alpha = 0.05
        z_alpha = stats.norm.ppf(0.975)
        
        # Power calculation
        z_beta = (abs(effect_size) - z_alpha * se) / se
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    def _calculate_risk_score(
        self,
        winner_data: Dict[str, Union[int, float]],
        control_data: Dict[str, Union[int, float]]
    ) -> float:
        """Calculate implementation risk score (0-1, higher = riskier)"""
        
        risk_factors = []
        
        # Sample size risk
        winner_n = winner_data.get('impressions', 0)
        if winner_n < 1000:
            risk_factors.append(0.3)
        elif winner_n < 5000:
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.0)
        
        # Conversion rate volatility risk
        winner_rate = winner_data.get('conversions', 0) / max(winner_n, 1)
        control_rate = control_data.get('conversions', 0) / max(control_data.get('impressions', 1), 1)
        
        relative_change = abs(winner_rate - control_rate) / max(control_rate, 0.01)
        if relative_change > 0.5:  # >50% change
            risk_factors.append(0.2)
        elif relative_change > 0.2:  # >20% change
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.0)
        
        # Confidence interval width risk
        ci = self._calculate_wilson_confidence_interval(
            int(winner_data.get('conversions', 0)), winner_n
        )
        ci_width = ci[1] - ci[0]
        
        if ci_width > 0.1:  # Wide confidence interval
            risk_factors.append(0.2)
        elif ci_width > 0.05:
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.0)
        
        return min(1.0, sum(risk_factors))
    
    def _assess_implementation_risks(
        self,
        winner_metrics: WinnerMetrics,
        validation_result: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks of implementing the winner"""
        
        risks = {
            'statistical_risks': [],
            'business_risks': [],
            'technical_risks': [],
            'overall_risk_level': 'low'
        }
        
        # Statistical risks
        if winner_metrics.statistical_power < 0.8:
            risks['statistical_risks'].append('Low statistical power - results may not replicate')
        
        if validation_result['confidence'] < 0.9:
            risks['statistical_risks'].append('Low confidence in winner validity')
        
        # Business risks
        if winner_metrics.business_value < 0:
            risks['business_risks'].append('Negative business impact detected')
        
        if business_context.get('implementation_cost', 0) > winner_metrics.business_value:
            risks['business_risks'].append('Implementation cost exceeds expected value')
        
        # Technical risks
        complexity = business_context.get('implementation_complexity', 'medium')
        if complexity == 'high':
            risks['technical_risks'].append('High implementation complexity')
        
        # Overall risk level
        total_risks = len(risks['statistical_risks']) + len(risks['business_risks']) + len(risks['technical_risks'])
        
        if total_risks >= 3:
            risks['overall_risk_level'] = 'high'
        elif total_risks >= 1:
            risks['overall_risk_level'] = 'medium'
        else:
            risks['overall_risk_level'] = 'low'
        
        return risks
    
    def _generate_recommendations(
        self,
        is_valid: bool,
        winner_metrics: WinnerMetrics,
        validation_result: Dict[str, Any],
        config: ValidationConfig,
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if is_valid:
            recommendations.append(f"✅ Winner validated - implement {winner_metrics.variant_id}")
            
            # Implementation guidance
            if risk_assessment['overall_risk_level'] == 'high':
                recommendations.append("⚠️ High risk detected - consider gradual rollout")
            elif risk_assessment['overall_risk_level'] == 'medium':
                recommendations.append("📊 Medium risk - monitor closely post-implementation")
            
            # Specific guidance based on metrics
            if winner_metrics.statistical_power < 0.8:
                recommendations.append("🔄 Consider running additional validation test")
            
            if validation_result['confidence'] > 0.95:
                recommendations.append("🚀 High confidence - safe for full rollout")
        
        else:
            recommendations.append("❌ Winner not validated - continue testing")
            
            # Specific improvement suggestions
            if winner_metrics.sample_size < config.min_sample_size:
                needed_samples = config.min_sample_size - winner_metrics.sample_size
                recommendations.append(f"📈 Collect {needed_samples:,} more samples")
            
            if abs(winner_metrics.effect_size) < config.min_effect_size:
                recommendations.append("🎯 Effect size too small - test larger changes")
            
            if winner_metrics.statistical_power < config.power_threshold:
                recommendations.append("⚡ Increase sample size to improve statistical power")
        
        # Method-specific recommendations
        if validation_result['method'] == 'hybrid':
            agreement = validation_result.get('methods_agreement', 0)
            if agreement == 1:
                recommendations.append("🤔 Statistical methods disagree - investigate further")
        
        return recommendations
    
    def get_validation_summary(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        
        if test_id not in self.validation_history:
            return {'error': 'No validation history found'}
        
        history = self.validation_history[test_id]
        latest = history[-1] if history else None
        
        summary = {
            'test_id': test_id,
            'total_validations': len(history),
            'current_status': 'validated' if latest and latest.is_valid_winner else 'not_validated',
            'validation_timeline': []
        }
        
        if latest:
            summary.update({
                'latest_validation': {
                    'is_valid': latest.is_valid_winner,
                    'confidence_score': latest.confidence_score,
                    'method': latest.validation_method,
                    'timestamp': latest.validation_timestamp
                },
                'winner_metrics': latest.business_metrics,
                'risk_level': latest.risk_assessment['overall_risk_level'],
                'recommendations': latest.recommendations
            })
        
        # Timeline of validations
        for i, validation in enumerate(history):
            summary['validation_timeline'].append({
                'validation_number': i + 1,
                'timestamp': validation.validation_timestamp,
                'result': 'valid' if validation.is_valid_winner else 'invalid',
                'confidence': validation.confidence_score,
                'method': validation.validation_method
            })
        
        return summary
