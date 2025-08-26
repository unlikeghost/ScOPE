# -*- coding: utf-8 -*-
"""
    ScOPE
    Dissimilarity Compression Metrics
    Jesus Alan Heernandez Galvan
"""
from typing import Optional

def safe_division(numerator: float, denominator: float, error_msg: str = "Division by zero") -> float:
    """Safely perform division with zero denominator check."""
    if denominator == 0:
        raise ZeroDivisionError(error_msg)
    return numerator / denominator

def ncd(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: Optional[int] = None) -> float:
    """Normalized Compression Dissimilarity"""
    _ = c_x2x1
    
    numerator = c_x1x2 - min(c_x1, c_x2)
    denominator = max(c_x1, c_x2)
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute NCD: max(c_x1, c_x2) is zero"
    )

def cdm(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: Optional[int] = None) -> float:
    """Compression Dissimilarity Measure"""
    _ = c_x2x1
    
    numerator = c_x1x2
    denominator = c_x1 + c_x2
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute CDM: c_x1 + c_x2 is zero"
    )
    
def cd(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: int) -> float:
    """Compression Dissimilarity"""
    numerator = min(c_x1, c_x2, c_x1x2, c_x2x1)
    denominator = c_x1 + c_x2
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute CD: c_x1 + c_x2 is zero"
    )

def ucd(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: int) -> float:
    """Universal Compression Dissimilarity"""
    numerator = max(c_x1x2 - c_x1, c_x2x1 - c_x2)
    denominator = max(c_x1, c_x2)
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute UCD: max(c_x1, c_x2) is zero"
    )

def ncc(c_x1: int, c_x2: int, c_x1x2: Optional[int] = None, c_x2x1: Optional[int] = None) -> float:
    """Normalized Conditional Compression"""
    _ = c_x1x2
    
    if c_x2x1 is None:
        raise ValueError("NCC requires c_x2x1 parameter")
    
    conditional_x1_x2 = c_x2x1 - c_x2  # C(x1|x2) approximation
    numerator = conditional_x1_x2
    denominator = c_x1
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute NCC: c_x1 is zero"
    )

def nccd(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: int) -> float:
    """Normalized Conditional Compression Dissimilarity"""
    conditional_x1_x2 = c_x2x1 - c_x2  # C(x1|x2) approximation
    conditional_x2_x1 = c_x1x2 - c_x1  # C(x2|x1) approximation
    
    numerator = max(conditional_x1_x2, conditional_x2_x1)
    denominator = max(c_x1, c_x2)
    
    return safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute NCCD: max(c_x1, c_x2) is zero"
    )

COMPRESSION_METRICS = {
    'ncd': ncd,
    'cdm': cdm,
    'cd': cd,
    'ucd': ucd,
    'ncc': ncc,
    'nccd': nccd
}

def compute_compression_metric(c_x1: int, c_x2: int, c_x1x2: int, c_x2x1: Optional[int], metric: str) -> float:
    """
    Compute the specified compression metric.
    
    Args:
        c_x1: Compression size of x1
        c_x2: Compression size of x2
        c_x1x2: Compression size of x1 concatenated with x2
        c_x2x1: Compression size of x2 concatenated with x1 (optional for some metrics)
        metric: Name of the metric to compute
        
    Returns:
        The computed metric value
        
    Raises:
        ValueError: If metric is not supported
    """
    metric_key = metric.lower()
    
    if metric_key not in COMPRESSION_METRICS:
        allowed = sorted(COMPRESSION_METRICS.keys())
        raise ValueError(f"'{metric}' is not a valid metric. Expected one of: {', '.join(allowed)}")
    
    metric_function = COMPRESSION_METRICS[metric_key]
    
    # Pass all parameters to maintain consistent interface
    return metric_function(c_x1, c_x2, c_x1x2, c_x2x1)