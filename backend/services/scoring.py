"""
Deal Scoring Service

This module provides hardened deal scoring functionality with comprehensive validation,
security controls, and error handling.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class DealStatus(Enum):
    """Enumeration of possible deal statuses."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class RiskLevel(Enum):
    """Enumeration of risk levels for deals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DealMetrics:
    """Data class for deal metrics with validation."""
    deal_id: str
    amount: Decimal
    probability: float
    days_in_pipeline: int
    customer_rating: float
    deal_status: DealStatus
    region: str
    product_category: str
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate all metrics are within acceptable ranges."""
        if self.amount <= 0:
            raise ValueError(f"Deal amount must be positive, got {self.amount}")
        
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")
        
        if self.days_in_pipeline < 0:
            raise ValueError(f"Days in pipeline cannot be negative, got {self.days_in_pipeline}")
        
        if not 0 <= self.customer_rating <= 5:
            raise ValueError(f"Customer rating must be between 0 and 5, got {self.customer_rating}")
        
        if not self.deal_id or not isinstance(self.deal_id, str):
            raise ValueError("Deal ID must be a non-empty string")
        
        if not self.region or not isinstance(self.region, str):
            raise ValueError("Region must be a non-empty string")
        
        if not self.product_category or not isinstance(self.product_category, str):
            raise ValueError("Product category must be a non-empty string")


@dataclass
class ScoringResult:
    """Data class for deal scoring results."""
    deal_id: str
    score: float
    risk_level: RiskLevel
    recommendation: str
    component_scores: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert scoring result to dictionary."""
        return {
            'deal_id': self.deal_id,
            'score': round(self.score, 2),
            'risk_level': self.risk_level.value,
            'recommendation': self.recommendation,
            'component_scores': {k: round(v, 2) for k, v in self.component_scores.items()},
            'timestamp': self.timestamp.isoformat()
        }


class DealScoringEngine:
    """Hardened deal scoring engine with multiple validation layers."""
    
    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        'amount': 0.25,
        'probability': 0.25,
        'pipeline_velocity': 0.20,
        'customer_quality': 0.20,
        'seasonality': 0.10
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        RiskLevel.LOW: (0.75, 1.0),
        RiskLevel.MEDIUM: (0.50, 0.75),
        RiskLevel.HIGH: (0.25, 0.50),
        RiskLevel.CRITICAL: (0.0, 0.25)
    }
    
    # Regional multipliers for seasonal adjustments
    REGIONAL_MULTIPLIERS = {
        'NORTH': 1.0,
        'SOUTH': 0.95,
        'EAST': 0.98,
        'WEST': 1.02,
        'CENTRAL': 0.97
    }
    
    # Product category multipliers
    CATEGORY_MULTIPLIERS = {
        'premium': 1.1,
        'standard': 1.0,
        'budget': 0.85,
        'enterprise': 1.15
    }
    
    def __init__(self, max_deal_amount: Decimal = Decimal('10000000')):
        """
        Initialize the scoring engine.
        
        Args:
            max_deal_amount: Maximum allowed deal amount for validation
        """
        self.max_deal_amount = max_deal_amount
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to 1.0."""
        total_weight = sum(self.WEIGHTS.values())
        if not abs(total_weight - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def score_deal(self, metrics: DealMetrics) -> ScoringResult:
        """
        Score a deal based on multiple factors with hardened validation.
        
        Args:
            metrics: DealMetrics object with deal information
            
        Returns:
            ScoringResult object with comprehensive scoring breakdown
            
        Raises:
            ValueError: If metrics validation fails
            TypeError: If metrics is not a DealMetrics instance
        """
        if not isinstance(metrics, DealMetrics):
            raise TypeError("Metrics must be a DealMetrics instance")
        
        # Check deal amount limits
        if metrics.amount > self.max_deal_amount:
            raise ValueError(f"Deal amount {metrics.amount} exceeds maximum {self.max_deal_amount}")
        
        # Calculate component scores
        component_scores = {
            'amount': self._score_amount(metrics.amount),
            'probability': self._score_probability(metrics.probability),
            'pipeline_velocity': self._score_pipeline_velocity(metrics.days_in_pipeline),
            'customer_quality': self._score_customer_quality(metrics.customer_rating),
            'seasonality': self._score_seasonality(metrics.region, metrics.product_category)
        }
        
        # Calculate weighted final score
        final_score = self._calculate_weighted_score(component_scores)
        
        # Apply status modifier
        final_score = self._apply_status_modifier(final_score, metrics.deal_status)
        
        # Ensure score is within valid range
        final_score = max(0.0, min(1.0, final_score))
        
        # Determine risk level and recommendation
        risk_level = self._determine_risk_level(final_score)
        recommendation = self._generate_recommendation(final_score, risk_level, metrics)
        
        result = ScoringResult(
            deal_id=metrics.deal_id,
            score=final_score,
            risk_level=risk_level,
            recommendation=recommendation,
            component_scores=component_scores,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Scored deal {metrics.deal_id}: score={final_score:.2f}, risk={risk_level.value}")
        
        return result
    
    def _score_amount(self, amount: Decimal) -> float:
        """
        Score based on deal amount using sigmoid normalization.
        
        Args:
            amount: Deal amount in currency units
            
        Returns:
            Score between 0 and 1
        """
        # Normalize to a reasonable scale (assume sweet spot around 500k)
        normalized = float(amount) / 500000.0
        # Apply sigmoid normalization
        score = 2.0 / (1.0 + (1.0 / (normalized + 0.1))) - 1.0
        return max(0.0, min(1.0, score))
    
    def _score_probability(self, probability: float) -> float:
        """
        Score based on deal probability.
        
        Args:
            probability: Closing probability (0-1)
            
        Returns:
            Score between 0 and 1
        """
        # Direct mapping with slight boost for high confidence
        if probability >= 0.8:
            return min(1.0, probability * 1.1)
        return probability
    
    def _score_pipeline_velocity(self, days_in_pipeline: int) -> float:
        """
        Score based on deal pipeline progression velocity.
        Higher velocity (fewer days) = better score.
        
        Args:
            days_in_pipeline: Number of days deal has been in pipeline
            
        Returns:
            Score between 0 and 1
        """
        # Optimal is 30-90 days; penalize very fast (potential red flag) or very slow
        if days_in_pipeline < 7:
            return 0.6  # Too fast may indicate incomplete process
        elif days_in_pipeline <= 90:
            return 1.0 - (days_in_pipeline / 200.0)
        else:
            return max(0.2, 1.0 - (days_in_pipeline / 100.0))
    
    def _score_customer_quality(self, customer_rating: float) -> float:
        """
        Score based on customer quality rating.
        
        Args:
            customer_rating: Customer rating (0-5)
            
        Returns:
            Score between 0 and 1
        """
        # Normalize 0-5 rating to 0-1 scale with emphasis on 4+ ratings
        normalized = customer_rating / 5.0
        if customer_rating >= 4:
            return min(1.0, normalized * 1.15)
        return normalized
    
    def _score_seasonality(self, region: str, product_category: str) -> float:
        """
        Score based on regional and seasonal factors.
        
        Args:
            region: Geographic region
            product_category: Product category
            
        Returns:
            Score between 0 and 1
        """
        # Get multipliers with safe defaults
        region_multiplier = self.REGIONAL_MULTIPLIERS.get(region.upper(), 1.0)
        category_multiplier = self.CATEGORY_MULTIPLIERS.get(product_category.lower(), 1.0)
        
        combined_score = (region_multiplier * category_multiplier) / 1.15
        return max(0.0, min(1.0, combined_score))
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """
        Calculate final score as weighted average of component scores.
        
        Args:
            component_scores: Dictionary of component scores
            
        Returns:
            Weighted score between 0 and 1
        """
        total = 0.0
        for component, weight in self.WEIGHTS.items():
            if component not in component_scores:
                raise ValueError(f"Missing component score: {component}")
            total += component_scores[component] * weight
        
        return total
    
    def _apply_status_modifier(self, score: float, status: DealStatus) -> float:
        """
        Apply modifiers based on deal status.
        
        Args:
            score: Current score
            status: Deal status
            
        Returns:
            Modified score
        """
        modifiers = {
            DealStatus.PENDING: 1.0,
            DealStatus.ACTIVE: 1.05,
            DealStatus.COMPLETED: 0.0,  # Completed deals are no longer scored
            DealStatus.CANCELLED: 0.0,
            DealStatus.ON_HOLD: 0.7
        }
        
        modifier = modifiers.get(status, 1.0)
        return score * modifier
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """
        Determine risk level based on score.
        
        Args:
            score: Deal score (0-1)
            
        Returns:
            RiskLevel enum value
        """
        for risk_level, (min_score, max_score) in self.RISK_THRESHOLDS.items():
            if min_score <= score < max_score:
                return risk_level
        
        return RiskLevel.CRITICAL
    
    def _generate_recommendation(
        self,
        score: float,
        risk_level: RiskLevel,
        metrics: DealMetrics
    ) -> str:
        """
        Generate actionable recommendation based on scoring.
        
        Args:
            score: Deal score
            risk_level: Risk level classification
            metrics: Original deal metrics
            
        Returns:
            Recommendation string
        """
        recommendations = []
        
        if risk_level == RiskLevel.LOW:
            recommendations.append("PROCEED: Strong deal with low risk. Prioritize for closure.")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("PROCEED WITH CAUTION: Monitor deal progress closely.")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("REVIEW: Deal requires additional due diligence before proceeding.")
        else:  # CRITICAL
            recommendations.append("HALT: Deal poses critical risk. Escalate to management.")
        
        # Add specific recommendations based on metrics
        if metrics.probability < 0.3:
            recommendations.append("- Low closing probability. Strengthen sales approach.")
        
        if metrics.days_in_pipeline > 180:
            recommendations.append("- Long pipeline duration. Identify and remove blockers.")
        
        if metrics.customer_rating < 2.5:
            recommendations.append("- Low customer rating. Conduct customer success review.")
        
        return " ".join(recommendations)
    
    def score_multiple_deals(
        self,
        metrics_list: List[DealMetrics]
    ) -> List[ScoringResult]:
        """
        Score multiple deals in batch.
        
        Args:
            metrics_list: List of DealMetrics objects
            
        Returns:
            List of ScoringResult objects
        """
        results = []
        failed_deals = []
        
        for metrics in metrics_list:
            try:
                result = self.score_deal(metrics)
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to score deal {metrics.deal_id}: {str(e)}")
                failed_deals.append((metrics.deal_id, str(e)))
        
        if failed_deals:
            logger.warning(f"Failed to score {len(failed_deals)} deals")
        
        return results


def create_scoring_engine(max_deal_amount: Decimal = Decimal('10000000')) -> DealScoringEngine:
    """
    Factory function to create a scoring engine instance.
    
    Args:
        max_deal_amount: Maximum allowed deal amount
        
    Returns:
        DealScoringEngine instance
    """
    return DealScoringEngine(max_deal_amount)
