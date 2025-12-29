"""
ABSA Data Extraction Package

A comprehensive module for extracting Aspect-Based Sentiment Analysis datasets.
"""

from .extraction import (
    # Main classes
    DatasetExtractor,
    SemEval2014Extractor,
    SemEval2015Extractor,
    SemEval2016Extractor,
    MultiDomainExtractor,

    # Data models
    Review,
    AspectTerm,
    AspectCategory,
    Opinion,

    # Convenience functions
    extract_semeval2014,
    extract_semeval2015,
    extract_semeval2016,
    extract_multidomain,
)

__version__ = '1.0.0'
__author__ = 'ABSA Data Extraction Team'

__all__ = [
    'DatasetExtractor',
    'SemEval2014Extractor',
    'SemEval2015Extractor',
    'SemEval2016Extractor',
    'MultiDomainExtractor',
    'Review',
    'AspectTerm',
    'AspectCategory',
    'Opinion',
    'extract_semeval2014',
    'extract_semeval2015',
    'extract_semeval2016',
    'extract_multidomain',
]
