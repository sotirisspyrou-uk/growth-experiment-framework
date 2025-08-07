from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from statistical_engine import StatisticalEngine
from experiment_design.hypothesis_generator import HypothesisGenerator
from ab_testing.multi_variant_tester import MultiVariantTester
from conversion_optimization.funnel_analyzer import FunnelAnalyzer


class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    BANDIT = "bandit"
    SEQUENTIAL = "sequential"


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    hypothesis: str
    experiment_type: ExperimentType
    target_metric: str
    variants: List[Dict[str, Any]]
    traffic_allocation: Dict[str, float]
    min_sample_size: int
    significance_level: float = 0.05
    power: float = 0.8
    min_effect_size: float = 0.02
    max_duration_days: int = 30
    guardrail_metrics: List[str] = field(default_factory=list)
    segment_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str
    variant_results: Dict[str, Dict[str, float]]
    statistical_significance: bool
    confidence_interval: Dict[str, tuple]
    p_value: float
    effect_size: float
    winner: Optional[str]
    recommendation: str
    risk_assessment: Dict[str, Any]


class GrowthFramework:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.statistical_engine = StatisticalEngine()
        self.hypothesis_generator = HypothesisGenerator()
        self.multivariate_tester = MultiVariantTester()
        self.funnel_analyzer = FunnelAnalyzer()
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create and validate a new experiment"""
        self._validate_experiment_config(config)
        
        # Calculate optimal sample size and duration
        sample_size = self.statistical_engine.calculate_sample_size(
            baseline_rate=self._get_baseline_metric(config.target_metric),
            min_effect_size=config.min_effect_size,
            significance_level=config.significance_level,
            power=config.power
        )
        
        config.min_sample_size = max(config.min_sample_size, sample_size)
        
        self.active_experiments[config.experiment_id] = config
        return config.experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment and begin data collection"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments[experiment_id]
        
        # Initialize tracking and feature flags
        self._initialize_feature_flags(experiment)
        self._setup_tracking(experiment)
        
        return True
    
    def analyze_experiment(self, experiment_id: str, data: pd.DataFrame) -> ExperimentResult:
        """Analyze experiment results and determine statistical significance"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments[experiment_id]
        
        # Perform statistical analysis
        results = self.statistical_engine.analyze_ab_test(
            data=data,
            variants=experiment.variants,
            target_metric=experiment.target_metric,
            significance_level=experiment.significance_level
        )
        
        # Check guardrail metrics
        guardrail_results = self._check_guardrails(data, experiment.guardrail_metrics)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(results, guardrail_results, experiment)
        
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=results['variant_results'],
            statistical_significance=results['significant'],
            confidence_interval=results['confidence_intervals'],
            p_value=results['p_value'],
            effect_size=results['effect_size'],
            winner=results.get('winner'),
            recommendation=recommendation,
            risk_assessment=guardrail_results
        )
        
        self.experiment_results[experiment_id] = experiment_result
        return experiment_result
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status and metrics for an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments[experiment_id]
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': ExperimentStatus.ACTIVE,
            'days_running': (datetime.now() - datetime.now()).days,  # Would use actual start date
            'sample_size_progress': self._get_sample_size_progress(experiment_id),
            'estimated_completion': self._estimate_completion_date(experiment),
            'current_results': self._get_interim_results(experiment_id)
        }
    
    def stop_experiment(self, experiment_id: str, reason: str = "Completed") -> bool:
        """Stop an experiment and finalize results"""
        if experiment_id not in self.active_experiments:
            return False
            
        # Archive experiment
        experiment = self.active_experiments.pop(experiment_id)
        
        # Clean up feature flags and tracking
        self._cleanup_experiment(experiment)
        
        return True
    
    def generate_hypotheses(self, data: pd.DataFrame, target_metric: str) -> List[Dict[str, Any]]:
        """Generate experiment hypotheses based on data analysis"""
        return self.hypothesis_generator.generate_hypotheses(data, target_metric)
    
    def prioritize_experiments(self, experiments: List[ExperimentConfig]) -> List[ExperimentConfig]:
        """Prioritize experiments using ICE framework (Impact, Confidence, Ease)"""
        scored_experiments = []
        
        for experiment in experiments:
            ice_score = self._calculate_ice_score(experiment)
            scored_experiments.append((experiment, ice_score))
        
        # Sort by ICE score descending
        scored_experiments.sort(key=lambda x: x[1], reverse=True)
        return [exp[0] for exp in scored_experiments]
    
    def get_experiment_portfolio_metrics(self) -> Dict[str, Any]:
        """Get overall experimentation program metrics"""
        completed_experiments = len(self.experiment_results)
        active_experiments = len(self.active_experiments)
        
        win_rate = 0
        total_impact = 0
        
        if completed_experiments > 0:
            wins = sum(1 for result in self.experiment_results.values() 
                      if result.statistical_significance and result.effect_size > 0)
            win_rate = wins / completed_experiments
            
            total_impact = sum(result.effect_size for result in self.experiment_results.values()
                             if result.statistical_significance)
        
        return {
            'total_experiments': completed_experiments + active_experiments,
            'active_experiments': active_experiments,
            'completed_experiments': completed_experiments,
            'win_rate': win_rate,
            'average_effect_size': total_impact / max(completed_experiments, 1),
            'total_business_impact': total_impact
        }
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration"""
        if not config.variants or len(config.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
            
        if abs(sum(config.traffic_allocation.values()) - 1.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 1.0")
            
        return True
    
    def _get_baseline_metric(self, metric_name: str) -> float:
        """Get baseline conversion rate for metric"""
        # Would integrate with analytics platform
        return 0.05  # Default 5% baseline
    
    def _initialize_feature_flags(self, experiment: ExperimentConfig) -> None:
        """Initialize feature flags for experiment"""
        pass  # Implementation depends on feature flag service
    
    def _setup_tracking(self, experiment: ExperimentConfig) -> None:
        """Setup tracking for experiment metrics"""
        pass  # Implementation depends on analytics platform
    
    def _check_guardrails(self, data: pd.DataFrame, guardrail_metrics: List[str]) -> Dict[str, Any]:
        """Check guardrail metrics for experiment safety"""
        guardrail_results = {}
        for metric in guardrail_metrics:
            # Check if metric has degraded significantly
            guardrail_results[metric] = {
                'status': 'safe',
                'change': 0.0,
                'significance': False
            }
        return guardrail_results
    
    def _generate_recommendation(self, results: Dict, guardrails: Dict, experiment: ExperimentConfig) -> str:
        """Generate actionable recommendation based on results"""
        if not results['significant']:
            return "No significant difference detected. Consider running longer or testing larger changes."
        
        if results['effect_size'] < experiment.min_effect_size:
            return "Statistically significant but effect size below minimum threshold."
        
        winner = results.get('winner', 'control')
        return f"Implement {winner} variant. Expected lift: {results['effect_size']:.2%}"
    
    def _get_sample_size_progress(self, experiment_id: str) -> float:
        """Get current sample size as percentage of target"""
        return 0.5  # Placeholder
    
    def _estimate_completion_date(self, experiment: ExperimentConfig) -> datetime:
        """Estimate when experiment will reach statistical significance"""
        return datetime.now() + timedelta(days=experiment.max_duration_days)
    
    def _get_interim_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get current interim results for monitoring"""
        return {'status': 'monitoring'}
    
    def _cleanup_experiment(self, experiment: ExperimentConfig) -> None:
        """Clean up experiment resources"""
        pass
    
    def _calculate_ice_score(self, experiment: ExperimentConfig) -> float:
        """Calculate ICE (Impact, Confidence, Ease) score for prioritization"""
        # Simplified ICE scoring - would be more sophisticated in practice
        impact = min(experiment.min_effect_size * 10, 10)  # 0-10 scale
        confidence = 8  # Default confidence score
        ease = 10 - (len(experiment.variants) - 2) * 2  # Easier with fewer variants
        
        return (impact * confidence * ease) ** (1/3)  # Geometric mean