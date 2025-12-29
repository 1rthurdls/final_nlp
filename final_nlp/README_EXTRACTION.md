# ABSA Data Extraction Module

A comprehensive Python module for extracting and processing Aspect-Based Sentiment Analysis (ABSA) datasets.

## Supported Datasets

1. **SemEval-2014 Task 4** - Laptop and Restaurant reviews
2. **SemEval-2015 Task 12** - Multi-domain sentiment analysis
3. **SemEval-2016 Task 5** - Aspect-based sentiment analysis
4. **Multi-Domain Aspect Extraction** - Custom datasets in JSON/CSV format

## Features

- ✅ Unified interface for multiple dataset formats
- ✅ Support for XML, JSON, and CSV formats
- ✅ Structured data models with type hints
- ✅ Comprehensive extraction of:
  - Aspect terms with positions
  - Aspect categories
  - Opinion expressions
  - Sentiment polarities
- ✅ Built-in statistics and analysis
- ✅ Easy export to JSON format
- ✅ Logging and error handling

## Installation

```bash
# No additional dependencies required beyond standard library
# Python 3.7+ recommended
```

## Quick Start

### Basic Usage

```python
from data.extraction import DatasetExtractor

# Initialize extractor
extractor = DatasetExtractor()

# Extract SemEval-2014 Restaurant data
reviews = extractor.extract(
    'data/semeval2014/Restaurants_Train.xml',
    dataset_type='semeval2014',
    domain='restaurant'
)

# Get statistics
stats = extractor.get_statistics(reviews)
print(f"Total reviews: {stats['total_reviews']}")
print(f"Total aspects: {stats['total_aspect_terms']}")

# Save to JSON
extractor.save_to_json(reviews, 'output/restaurant_reviews.json')
```

### Using Convenience Functions

```python
from data.extraction import (
    extract_semeval2014,
    extract_semeval2015,
    extract_semeval2016,
    extract_multidomain
)

# SemEval-2014
restaurant_reviews = extract_semeval2014(
    'data/semeval2014/Restaurants_Train.xml',
    domain='restaurant'
)

laptop_reviews = extract_semeval2014(
    'data/semeval2014/Laptop_Train.xml',
    domain='laptop'
)

# SemEval-2015
reviews_2015 = extract_semeval2015(
    'data/semeval2015/ABSA15_RestaurantsTrain.xml',
    domain='restaurant'
)

# SemEval-2016
reviews_2016 = extract_semeval2016(
    'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
    domain='restaurant'
)

# Multi-domain (JSON/CSV)
multi_reviews = extract_multidomain(
    'data/multidomain/electronics.json',
    domain='electronics'
)
```

## Data Models

### Review Object

```python
@dataclass
class Review:
    review_id: str                          # Unique identifier
    text: str                               # Review text
    aspect_terms: List[AspectTerm]          # Aspect terms with positions
    aspect_categories: List[AspectCategory] # Aspect categories
    opinions: List[Opinion]                 # Opinion expressions
    domain: Optional[str]                   # Domain (e.g., 'restaurant')
```

### AspectTerm Object

```python
@dataclass
class AspectTerm:
    term: str                    # Aspect term text
    polarity: str                # Sentiment polarity
    from_index: Optional[int]    # Start position in text
    to_index: Optional[int]      # End position in text
```

### AspectCategory Object

```python
@dataclass
class AspectCategory:
    category: str    # Category (e.g., 'FOOD#QUALITY')
    polarity: str    # Sentiment polarity
```

### Opinion Object

```python
@dataclass
class Opinion:
    target: Optional[str]          # Target expression
    target_from: Optional[int]     # Target start position
    target_to: Optional[int]       # Target end position
    category: Optional[str]        # Aspect category
    polarity: Optional[str]        # Sentiment polarity
    expression: Optional[str]      # Opinion expression
    expression_from: Optional[int] # Expression start position
    expression_to: Optional[int]   # Expression end position
```

## Dataset-Specific Examples

### SemEval-2014 Task 4

Expected XML structure:
```xml
<sentences>
  <sentence id="1">
    <text>The food was delicious but the service was slow.</text>
    <aspectTerms>
      <aspectTerm term="food" polarity="positive" from="4" to="8"/>
      <aspectTerm term="service" polarity="negative" from="33" to="40"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="FOOD#QUALITY" polarity="positive"/>
      <aspectCategory category="SERVICE#GENERAL" polarity="negative"/>
    </aspectCategories>
  </sentence>
</sentences>
```

### SemEval-2015/2016 Task

Expected XML structure:
```xml
<Reviews>
  <Review rid="123">
    <sentences>
      <sentence id="123:0">
        <text>Great food, terrible atmosphere.</text>
        <Opinions>
          <Opinion target="food" category="FOOD#QUALITY" polarity="positive" from="6" to="10"/>
          <Opinion target="atmosphere" category="AMBIENCE#GENERAL" polarity="negative" from="21" to="31"/>
        </Opinions>
      </sentence>
    </sentences>
  </Review>
</Reviews>
```

### Multi-Domain JSON

Expected JSON structure:
```json
[
  {
    "id": "review_1",
    "text": "The battery life is amazing.",
    "domain": "electronics",
    "aspects": [
      {
        "term": "battery life",
        "polarity": "positive",
        "from": 4,
        "to": 16
      }
    ]
  }
]
```

### Multi-Domain CSV

Expected CSV structure:
```csv
id,text,aspect,polarity,domain
1,"Great phone with excellent camera",camera,positive,electronics
2,"Screen quality is poor",screen,negative,electronics
```

## Advanced Usage

### Batch Processing Multiple Datasets

```python
from data.extraction import DatasetExtractor

extractor = DatasetExtractor()

datasets = [
    ('data/semeval2014/Restaurants_Train.xml', 'semeval2014', 'restaurant'),
    ('data/semeval2015/ABSA15_RestaurantsTrain.xml', 'semeval2015', 'restaurant'),
    ('data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml', 'semeval2016', 'restaurant'),
]

all_reviews = []
for path, dataset_type, domain in datasets:
    reviews = extractor.extract(path, dataset_type=dataset_type, domain=domain)
    all_reviews.extend(reviews)
    print(f"Loaded {len(reviews)} from {dataset_type}")

# Combined statistics
stats = extractor.get_statistics(all_reviews)
print(f"Total reviews across all datasets: {stats['total_reviews']}")
```

### Custom Data Processing

```python
from data.extraction import extract_semeval2014

reviews = extract_semeval2014('data/semeval2014/Restaurants_Train.xml')

# Extract all aspect-sentiment pairs
aspect_sentiment_pairs = []
for review in reviews:
    for aspect in review.aspect_terms:
        aspect_sentiment_pairs.append({
            'text': review.text,
            'aspect': aspect.term,
            'sentiment': aspect.polarity,
            'context': review.text[max(0, aspect.from_index-20):aspect.to_index+20]
        })

# Analyze polarity distribution
from collections import Counter
polarity_counts = Counter(pair['sentiment'] for pair in aspect_sentiment_pairs)
print(polarity_counts)
```

### Working with Multi-Domain CSV

```python
from data.extraction import MultiDomainExtractor

extractor = MultiDomainExtractor()

# Custom CSV with specific columns
reviews = extractor.extract_from_csv(
    'data/custom_reviews.csv',
    domain='electronics',
    text_column='review_text',      # Column containing review text
    aspect_column='product_feature', # Column containing aspect
    polarity_column='sentiment'      # Column containing sentiment
)
```

## Statistics and Analysis

The module provides comprehensive statistics:

```python
stats = extractor.get_statistics(reviews)

# Returns:
{
    'total_reviews': 3000,
    'total_aspect_terms': 4500,
    'total_aspect_categories': 3500,
    'total_opinions': 4200,
    'domains': ['restaurant', 'laptop'],
    'polarity_distribution': {
        'positive': 2000,
        'negative': 1500,
        'neutral': 1000
    },
    'avg_aspect_terms_per_review': 1.5,
    'avg_opinions_per_review': 1.4
}
```

## File Organization

```
final_nlp/
├── data/
│   ├── extraction.py              # Main extraction module
│   ├── semeval2014/              # SemEval-2014 datasets
│   │   ├── Restaurants_Train.xml
│   │   └── Laptop_Train.xml
│   ├── semeval2015/              # SemEval-2015 datasets
│   │   └── ABSA15_RestaurantsTrain.xml
│   ├── semeval2016/              # SemEval-2016 datasets
│   │   └── ABSA16_Restaurants_Train_SB1_v2.xml
│   └── multidomain/              # Multi-domain datasets
│       ├── electronics.json
│       └── products.csv
├── output/                        # Extracted JSON files
├── example_extraction.py          # Usage examples
└── README_EXTRACTION.md          # This file
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    reviews = extractor.extract(
        'data/nonexistent.xml',
        dataset_type='semeval2014',
        domain='restaurant'
    )
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid dataset type or format: {e}")
```

## Logging

The module uses Python's logging module:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Extract data (will show detailed logs)
reviews = extractor.extract(...)
```

## Common Dataset Sources

- **SemEval-2014**: http://alt.qcri.org/semeval2014/task4/
- **SemEval-2015**: http://alt.qcri.org/semeval2015/task12/
- **SemEval-2016**: http://alt.qcri.org/semeval2016/task5/

## Troubleshooting

### Issue: XML parsing errors

**Solution**: Ensure your XML files are well-formed and follow the expected schema for each dataset type.

### Issue: Empty aspect lists

**Solution**: Some reviews may not have annotated aspects. This is normal. Filter them if needed:

```python
reviews_with_aspects = [r for r in reviews if r.aspect_terms or r.opinions]
```

### Issue: Encoding errors

**Solution**: The module uses UTF-8 encoding by default. Ensure your files are UTF-8 encoded.

## Contributing

To extend the module for additional datasets:

1. Create a new extractor class inheriting from a base structure
2. Implement `extract_from_xml()`, `extract_from_json()`, or `extract_from_csv()`
3. Add to the `DatasetExtractor.extractors` dictionary

## License

This module is provided as-is for research and educational purposes.

## Citation

If you use this module in your research, please cite the original SemEval tasks:

```bibtex
@inproceedings{pontiki2014semeval,
  title={SemEval-2014 Task 4: Aspect Based Sentiment Analysis},
  author={Pontiki, Maria and Galanis, Dimitrios and Pavlopoulos, John and Papageorgiou, Harris and Androutsopoulos, Ion and Manandhar, Suresh},
  booktitle={Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)},
  year={2014}
}
```
