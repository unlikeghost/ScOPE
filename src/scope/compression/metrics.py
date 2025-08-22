# -*- coding: utf-8 -*-

def safe_division(numerator: float, denominator: float, error_msg: str = "Division by zero") -> float:
   if denominator == 0:
       raise ZeroDivisionError(error_msg)
   return numerator / denominator


def ncd(c_x1: int, c_x2: int, c_x1x2: int) -> float:
   numerator = c_x1x2 - min(c_x1, c_x2)
   denominator = max(c_x1, c_x2)
   return safe_division(numerator, denominator, "Cannot compute NCD: max(c_x1, c_x2) is zero")


def cdm(c_x1: int, c_x2: int, c_x1x2: int) -> float:
   numerator = c_x1x2
   denominator = c_x1 + c_x2
   return safe_division(numerator, denominator, "Cannot compute CDM: c_x1 + c_x2 is zero")


def nrc(c_x1: int, c_x2: int, c_x1x2: int) -> float:
   numerator = c_x1x2
   denominator = c_x1
   return safe_division(numerator, denominator, "Cannot compute NRC: c_x1 is zero")


COMPRESSION_METRICS = {
   'ncd': ncd,
   'cdm': cdm,
   'nrc': nrc
}


def compute_compression_metric(c_x1: int, c_x2: int, c_x1x2: int, metric: str) -> float:
   metric_key = metric.lower()
   
   if metric_key not in COMPRESSION_METRICS:
       allowed = sorted(COMPRESSION_METRICS.keys())
       raise ValueError(f"'{metric}' is not a valid metric. Expected one of: {', '.join(allowed)}")
   
   metric_function = COMPRESSION_METRICS[metric_key]
   return metric_function(c_x1, c_x2, c_x1x2)