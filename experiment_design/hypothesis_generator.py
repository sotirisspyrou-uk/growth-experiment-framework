import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class HypothesisType(Enum):
    CONVERSION_OPTIMIZATION = "conversion_optimization"
    USER_ENGAGEMENT = "user_engagement"
    RETENTION_IMPROVEMENT = "retention_improvement"
    REVENUE_GROWTH = "revenue_growth"
    USER_EXPERIENCE = "user_experience"


class ImpactLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class HypothesisInsight:
    hypothesis_id: str
    hypothesis_text: str
    hypothesis_type: HypothesisType
    expected_impact: ImpactLevel
    confidence_score: float
    supporting_data: Dict[str, Any]
    suggested_variants: List[Dict[str, str]]
    success_metrics: List[str]
    implementation_effort: str
    ice_score: float


@dataclass
class DataInsight:
    insight_type: str
    description: str
    statistical_significance: float
    effect_size: float
    supporting_metrics: Dict[str, float]


class HypothesisGenerator:
    """AI-powered hypothesis generation based on data patterns and growth opportunities"""
    
    def __init__(self):
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.scaler = StandardScaler()
        
    def generate_hypotheses(
        self, 
        data: pd.DataFrame, 
        target_metric: str,
        business_context: Dict[str, Any] = None,
        max_hypotheses: int = 10
    ) -> List[HypothesisInsight]:
        """Generate experiment hypotheses based on data analysis"""
        
        # Analyze data patterns
        insights = self._analyze_data_patterns(data, target_metric)
        
        # Generate hypotheses from insights
        hypotheses = []
        for insight in insights:
            hypothesis_list = self._generate_hypotheses_from_insight(
                insight, data, target_metric, business_context or {}
            )
            hypotheses.extend(hypothesis_list)
        
        # Score and rank hypotheses
        scored_hypotheses = self._score_hypotheses(hypotheses, data)
        
        # Return top hypotheses
        return sorted(scored_hypotheses, key=lambda x: x.ice_score, reverse=True)[:max_hypotheses]
    
    def _analyze_data_patterns(self, data: pd.DataFrame, target_metric: str) -> List[DataInsight]:
        """Analyze data to identify patterns and opportunities"""
        insights = []
        
        # Segment analysis
        segment_insights = self._analyze_segments(data, target_metric)
        insights.extend(segment_insights)
        
        # Conversion funnel analysis
        funnel_insights = self._analyze_conversion_funnel(data, target_metric)
        insights.extend(funnel_insights)
        
        # Time-based patterns
        temporal_insights = self._analyze_temporal_patterns(data, target_metric)
        insights.extend(temporal_insights)
        
        # Device/platform analysis
        device_insights = self._analyze_device_patterns(data, target_metric)
        insights.extend(device_insights)
        
        return insights
    
    def _analyze_segments(self, data: pd.DataFrame, target_metric: str) -> List[DataInsight]:
        """Analyze user segments for optimization opportunities"""
        insights = []
        
        # Identify potential segmentation columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col == target_metric or data[col].nunique() > 20:
                continue
                
            # Calculate conversion rates by segment
            segment_performance = data.groupby(col)[target_metric].agg(['mean', 'count']).reset_index()
            segment_performance = segment_performance[segment_performance['count'] >= 30]  # Minimum sample size
            
            if len(segment_performance) < 2:
                continue
            
            # Find significant differences
            best_segment = segment_performance.loc[segment_performance['mean'].idxmax()]
            worst_segment = segment_performance.loc[segment_performance['mean'].idxmin()]
            
            # Calculate statistical significance
            best_data = data[data[col] == best_segment[col]][target_metric]
            worst_data = data[data[col] == worst_segment[col]][target_metric]
            
            if len(best_data) > 0 and len(worst_data) > 0:
                _, p_value = stats.ttest_ind(best_data, worst_data)
                effect_size = (best_segment['mean'] - worst_segment['mean']) / np.sqrt(
                    (best_data.var() + worst_data.var()) / 2
                )
                
                if p_value < 0.05 and abs(effect_size) > 0.2:  # Medium effect size
                    insights.append(DataInsight(
                        insight_type="segment_opportunity",
                        description=f"Segment '{best_segment[col]}' performs {(best_segment['mean']/worst_segment['mean']-1)*100:.1f}% better than '{worst_segment[col]}'",
                        statistical_significance=1 - p_value,
                        effect_size=effect_size,
                        supporting_metrics={
                            'best_segment_rate': float(best_segment['mean']),
                            'worst_segment_rate': float(worst_segment['mean']),
                            'segment_column': col
                        }
                    ))
        
        return insights
    
    def _analyze_conversion_funnel(self, data: pd.DataFrame, target_metric: str) -> List[DataInsight]:
        """Analyze conversion funnel for bottlenecks"""
        insights = []
        
        # Identify funnel steps (columns that look like sequential events)
        funnel_cols = [col for col in data.columns if 
                      col.startswith(('step_', 'stage_', 'visited_', 'completed_')) or
                      col.endswith(('_page', '_view', '_click', '_complete'))]
        
        if len(funnel_cols) >= 2:
            funnel_data = data[funnel_cols + [target_metric]].copy()
            
            # Calculate conversion rates between steps
            conversion_rates = {}
            for i, col in enumerate(funnel_cols):
                if i == 0:
                    conversion_rates[col] = funnel_data[col].mean()
                else:
                    # Conditional conversion rate
                    prev_col = funnel_cols[i-1]
                    completed_previous = funnel_data[funnel_data[prev_col] == 1]
                    if len(completed_previous) > 0:
                        conversion_rates[f"{prev_col}_to_{col}"] = completed_previous[col].mean()
            
            # Find the biggest drop-off
            if conversion_rates:
                min_conversion_step = min(conversion_rates.items(), key=lambda x: x[1])
                
                insights.append(DataInsight(
                    insight_type="funnel_bottleneck",
                    description=f"Major conversion bottleneck at step '{min_conversion_step[0]}' with {min_conversion_step[1]:.1%} conversion rate",
                    statistical_significance=0.95,  # High confidence in funnel analysis
                    effect_size=1.0 - min_conversion_step[1],  # Drop-off rate as effect size
                    supporting_metrics={
                        'bottleneck_step': min_conversion_step[0],
                        'conversion_rate': min_conversion_step[1],
                        'all_step_rates': conversion_rates
                    }
                ))
        
        return insights
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame, target_metric: str) -> List[DataInsight]:
        """Analyze time-based patterns in conversions"""
        insights = []
        
        # Look for date/time columns
        date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        
        for date_col in date_cols:
            # Hour of day analysis
            if 'hour' not in data.columns:
                data['hour'] = pd.to_datetime(data[date_col]).dt.hour
            
            hourly_performance = data.groupby('hour')[target_metric].agg(['mean', 'count'])
            hourly_performance = hourly_performance[hourly_performance['count'] >= 10]
            
            if len(hourly_performance) > 0:
                best_hour = hourly_performance['mean'].idxmax()
                worst_hour = hourly_performance['mean'].idxmin()
                
                improvement_potential = (
                    hourly_performance.loc[best_hour, 'mean'] - 
                    hourly_performance.loc[worst_hour, 'mean']
                ) / hourly_performance.loc[worst_hour, 'mean']
                
                if improvement_potential > 0.2:  # 20% improvement potential
                    insights.append(DataInsight(
                        insight_type="temporal_opportunity",
                        description=f"Hour {best_hour} performs {improvement_potential:.1%} better than hour {worst_hour}",
                        statistical_significance=0.85,
                        effect_size=improvement_potential,
                        supporting_metrics={
                            'best_hour': int(best_hour),
                            'worst_hour': int(worst_hour),
                            'improvement_potential': improvement_potential,
                            'hourly_data': hourly_performance.to_dict()
                        }
                    ))
        
        return insights
    
    def _analyze_device_patterns(self, data: pd.DataFrame, target_metric: str) -> List[DataInsight]:
        """Analyze device and platform performance differences"""
        insights = []
        
        device_cols = [col for col in data.columns if 
                      col.lower() in ['device', 'device_type', 'platform', 'browser', 'os']]
        
        for col in device_cols:
            if data[col].nunique() < 2:
                continue
                
            device_performance = data.groupby(col)[target_metric].agg(['mean', 'count']).reset_index()
            device_performance = device_performance[device_performance['count'] >= 20]
            
            if len(device_performance) >= 2:
                best_device = device_performance.loc[device_performance['mean'].idxmax()]
                worst_device = device_performance.loc[device_performance['mean'].idxmin()]
                
                improvement = (best_device['mean'] - worst_device['mean']) / worst_device['mean']
                
                if improvement > 0.15:  # 15% improvement opportunity
                    insights.append(DataInsight(
                        insight_type="device_opportunity",
                        description=f"{best_device[col]} converts {improvement:.1%} better than {worst_device[col]}",
                        statistical_significance=0.80,
                        effect_size=improvement,
                        supporting_metrics={
                            'best_device': str(best_device[col]),
                            'worst_device': str(worst_device[col]),
                            'device_column': col,
                            'improvement_potential': improvement
                        }
                    ))
        
        return insights
    
    def _generate_hypotheses_from_insight(
        self, 
        insight: DataInsight, 
        data: pd.DataFrame,
        target_metric: str,
        business_context: Dict[str, Any]
    ) -> List[HypothesisInsight]:
        """Generate specific hypotheses from data insights"""
        hypotheses = []
        
        if insight.insight_type == "segment_opportunity":
            hypotheses.extend(self._generate_segment_hypotheses(insight, business_context))
        elif insight.insight_type == "funnel_bottleneck":
            hypotheses.extend(self._generate_funnel_hypotheses(insight, business_context))
        elif insight.insight_type == "temporal_opportunity":
            hypotheses.extend(self._generate_temporal_hypotheses(insight, business_context))
        elif insight.insight_type == "device_opportunity":
            hypotheses.extend(self._generate_device_hypotheses(insight, business_context))
        
        return hypotheses
    
    def _generate_segment_hypotheses(self, insight: DataInsight, context: Dict) -> List[HypothesisInsight]:
        """Generate hypotheses for segment optimization"""
        segment_col = insight.supporting_metrics['segment_column']
        best_rate = insight.supporting_metrics['best_segment_rate']
        worst_rate = insight.supporting_metrics['worst_segment_rate']
        
        return [
            HypothesisInsight(
                hypothesis_id=f"segment_optimization_{segment_col}",
                hypothesis_text=f"By optimizing the experience for the underperforming {segment_col} segment, we can increase conversion rates from {worst_rate:.1%} to {best_rate:.1%}",
                hypothesis_type=HypothesisType.CONVERSION_OPTIMIZATION,
                expected_impact=ImpactLevel.HIGH if insight.effect_size > 0.5 else ImpactLevel.MEDIUM,
                confidence_score=insight.statistical_significance,
                supporting_data=insight.supporting_metrics,
                suggested_variants=[
                    {"name": "control", "description": "Current experience"},
                    {"name": "optimized", "description": f"Experience optimized for underperforming {segment_col}"}
                ],
                success_metrics=[target_metric, f"{segment_col}_conversion_rate"],
                implementation_effort="medium",
                ice_score=0.0  # Will be calculated later
            )
        ]
    
    def _generate_funnel_hypotheses(self, insight: DataInsight, context: Dict) -> List[HypothesisInsight]:
        """Generate hypotheses for funnel optimization"""
        bottleneck = insight.supporting_metrics['bottleneck_step']
        current_rate = insight.supporting_metrics['conversion_rate']
        
        return [
            HypothesisInsight(
                hypothesis_id=f"funnel_optimization_{bottleneck}",
                hypothesis_text=f"By removing friction at the {bottleneck} step, we can increase conversion from {current_rate:.1%} to at least {(current_rate * 1.3):.1%}",
                hypothesis_type=HypothesisType.CONVERSION_OPTIMIZATION,
                expected_impact=ImpactLevel.HIGH,
                confidence_score=0.85,
                supporting_data=insight.supporting_metrics,
                suggested_variants=[
                    {"name": "control", "description": "Current funnel"},
                    {"name": "simplified", "description": f"Simplified {bottleneck} step"},
                    {"name": "enhanced", "description": f"Enhanced {bottleneck} with better UX"}
                ],
                success_metrics=[target_metric, "funnel_completion_rate"],
                implementation_effort="high",
                ice_score=0.0
            )
        ]
    
    def _generate_temporal_hypotheses(self, insight: DataInsight, context: Dict) -> List[HypothesisInsight]:
        """Generate hypotheses for time-based optimization"""
        best_hour = insight.supporting_metrics['best_hour']
        worst_hour = insight.supporting_metrics['worst_hour']
        
        return [
            HypothesisInsight(
                hypothesis_id=f"temporal_optimization_timing",
                hypothesis_text=f"By adjusting messaging/offers to match the high-performing patterns from hour {best_hour}, we can improve conversion during low-performing times like hour {worst_hour}",
                hypothesis_type=HypothesisType.USER_ENGAGEMENT,
                expected_impact=ImpactLevel.MEDIUM,
                confidence_score=0.75,
                supporting_data=insight.supporting_metrics,
                suggested_variants=[
                    {"name": "control", "description": "Standard messaging"},
                    {"name": "time_optimized", "description": "Messaging optimized for time of day"}
                ],
                success_metrics=[target_metric, "hourly_conversion_variance"],
                implementation_effort="low",
                ice_score=0.0
            )
        ]
    
    def _generate_device_hypotheses(self, insight: DataInsight, context: Dict) -> List[HypothesisInsight]:
        """Generate hypotheses for device-specific optimization"""
        device_col = insight.supporting_metrics['device_column']
        worst_device = insight.supporting_metrics['worst_device']
        
        return [
            HypothesisInsight(
                hypothesis_id=f"device_optimization_{device_col}",
                hypothesis_text=f"By creating a {worst_device}-optimized experience, we can close the performance gap and increase overall conversion rates",
                hypothesis_type=HypothesisType.USER_EXPERIENCE,
                expected_impact=ImpactLevel.HIGH,
                confidence_score=0.80,
                supporting_data=insight.supporting_metrics,
                suggested_variants=[
                    {"name": "control", "description": "Current responsive design"},
                    {"name": "device_optimized", "description": f"Experience specifically optimized for {worst_device}"}
                ],
                success_metrics=[target_metric, f"{device_col}_conversion_parity"],
                implementation_effort="high",
                ice_score=0.0
            )
        ]
    
    def _score_hypotheses(self, hypotheses: List[HypothesisInsight], data: pd.DataFrame) -> List[HypothesisInsight]:
        """Score hypotheses using ICE framework (Impact, Confidence, Ease)"""
        for hypothesis in hypotheses:
            # Impact score (1-10)
            impact_score = hypothesis.expected_impact.value * 3
            if hypothesis.hypothesis_type in [HypothesisType.CONVERSION_OPTIMIZATION, HypothesisType.REVENUE_GROWTH]:
                impact_score += 1
            
            # Confidence score (1-10)
            confidence_score = min(hypothesis.confidence_score * 10, 10)
            
            # Ease score (1-10) - inverse of implementation effort
            ease_mapping = {"low": 8, "medium": 5, "high": 2}
            ease_score = ease_mapping.get(hypothesis.implementation_effort, 5)
            
            # Calculate ICE score (geometric mean)
            hypothesis.ice_score = (impact_score * confidence_score * ease_score) ** (1/3)
        
        return hypotheses
    
    def _load_hypothesis_templates(self) -> Dict[str, List[str]]:
        """Load hypothesis templates for different scenarios"""
        return {
            "conversion_optimization": [
                "By {change}, we can increase conversion rate from {baseline} to {target}",
                "Improving {element} will reduce friction and increase conversions by {expected_lift}",
                "Users will convert more when we {intervention} because {reasoning}"
            ],
            "user_engagement": [
                "By {change}, we can increase user engagement metrics by {expected_lift}",
                "Users will be more engaged when {condition} because {reasoning}"
            ],
            "retention_improvement": [
                "By {intervention}, we can improve user retention from {baseline} to {target}",
                "Users will stay longer when we {change} because {reasoning}"
            ]
        }
    
    def prioritize_hypotheses_by_business_impact(
        self,
        hypotheses: List[HypothesisInsight],
        business_priorities: Dict[str, float]
    ) -> List[HypothesisInsight]:
        """Prioritize hypotheses based on business priorities"""
        for hypothesis in hypotheses:
            # Adjust ICE score based on business priorities
            business_multiplier = business_priorities.get(hypothesis.hypothesis_type.value, 1.0)
            hypothesis.ice_score *= business_multiplier
        
        return sorted(hypotheses, key=lambda x: x.ice_score, reverse=True)
    
    def generate_experiment_variants(
        self,
        hypothesis: HypothesisInsight,
        max_variants: int = 4
    ) -> List[Dict[str, Any]]:
        """Generate detailed experiment variants for a hypothesis"""
        base_variants = hypothesis.suggested_variants
        
        # Enhance variants with more details
        enhanced_variants = []
        for i, variant in enumerate(base_variants[:max_variants]):
            enhanced_variants.append({
                "id": f"{hypothesis.hypothesis_id}_variant_{i}",
                "name": variant["name"],
                "description": variant["description"],
                "implementation_details": self._generate_implementation_details(
                    hypothesis, variant
                ),
                "success_criteria": hypothesis.success_metrics,
                "estimated_impact": self._estimate_variant_impact(hypothesis, variant)
            })
        
        return enhanced_variants
    
    def _generate_implementation_details(
        self, 
        hypothesis: HypothesisInsight, 
        variant: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate implementation details for a variant"""
        # This would be more sophisticated in practice
        return {
            "ui_changes": f"Update interface for {variant['name']}",
            "backend_changes": "Minimal backend changes required",
            "tracking_requirements": hypothesis.success_metrics,
            "estimated_dev_hours": {"low": 8, "medium": 40, "high": 120}.get(
                hypothesis.implementation_effort, 40
            )
        }
    
    def _estimate_variant_impact(
        self, 
        hypothesis: HypothesisInsight, 
        variant: Dict[str, str]
    ) -> Dict[str, float]:
        """Estimate the potential impact of a variant"""
        base_impact = hypothesis.expected_impact.value / 3.0  # Convert to decimal
        
        # Control variant has no impact
        if variant["name"] == "control":
            return {"expected_lift": 0.0, "confidence_interval": (0.0, 0.0)}
        
        # Estimate based on hypothesis confidence and impact
        expected_lift = base_impact * hypothesis.confidence_score
        ci_lower = expected_lift * 0.5
        ci_upper = expected_lift * 1.5
        
        return {
            "expected_lift": expected_lift,
            "confidence_interval": (ci_lower, ci_upper)
        }