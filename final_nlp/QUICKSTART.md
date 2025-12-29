# Quick Start Guide - ABSA Data Extraction Module

## Installation

No installation required! The module uses only Python standard library.

```bash
# Just ensure you have Python 3.7+
python --version
```

## Basic Usage (5 minutes)

### 1. Extract SemEval-2014 Data

```python
from data.extraction import extract_semeval2014

# Extract restaurant reviews
reviews = extract_semeval2014(
    'data/semeval2014/Restaurants_Train.xml',
    domain='restaurant'
)

print(f"Loaded {len(reviews)} reviews")

# Access data
for review in reviews[:3]:  # First 3 reviews
    print(f"\nReview: {review.text}")
    for aspect in review.aspect_terms:
        print(f"  - {aspect.term}: {aspect.polarity}")
```

### 2. Extract SemEval-2015 Data

```python
from data.extraction import extract_semeval2015

reviews = extract_semeval2015(
    'data/semeval2015/ABSA15_RestaurantsTrain.xml',
    domain='restaurant'
)

# Access opinions
for review in reviews[:3]:
    for opinion in review.opinions:
        print(f"{opinion.target} -> {opinion.category}: {opinion.polarity}")
```

### 3. Extract SemEval-2016 Data

```python
from data.extraction import extract_semeval2016

reviews = extract_semeval2016(
    'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
    domain='restaurant'
)
```

### 4. Extract Multi-Domain Data (JSON/CSV)

```python
from data.extraction import extract_multidomain

# From JSON
reviews = extract_multidomain(
    'data/multidomain/electronics.json',
    domain='electronics'
)

# From CSV
from data.extraction import MultiDomainExtractor
extractor = MultiDomainExtractor()
reviews = extractor.extract_from_csv(
    'data/products.csv',
    domain='products',
    text_column='review_text',
    aspect_column='feature',
    polarity_column='sentiment'
)
```

### 5. Save to JSON

```python
from data.extraction import DatasetExtractor

extractor = DatasetExtractor()

# Extract data
reviews = extractor.extract(
    'data/semeval2014/Restaurants_Train.xml',
    dataset_type='semeval2014',
    domain='restaurant'
)

# Save to JSON
extractor.save_to_json(reviews, 'output/restaurant_data.json')
```

### 6. Get Statistics

```python
from data.extraction import DatasetExtractor

extractor = DatasetExtractor()
reviews = extractor.extract(
    'data/semeval2014/Restaurants_Train.xml',
    dataset_type='semeval2014',
    domain='restaurant'
)

# Get stats
stats = extractor.get_statistics(reviews)
print(f"Total reviews: {stats['total_reviews']}")
print(f"Total aspects: {stats['total_aspect_terms']}")
print(f"Polarity distribution: {stats['polarity_distribution']}")
```

## Common Patterns

### Pattern 1: Batch Process Multiple Datasets

```python
from data.extraction import DatasetExtractor

extractor = DatasetExtractor()

datasets = [
    ('data/semeval2014/Restaurants_Train.xml', 'semeval2014', 'restaurant'),
    ('data/semeval2015/ABSA15_RestaurantsTrain.xml', 'semeval2015', 'restaurant'),
]

for path, dataset_type, domain in datasets:
    reviews = extractor.extract(path, dataset_type, domain)
    output = f'output/{dataset_type}_{domain}.json'
    extractor.save_to_json(reviews, output)
    print(f"Saved {len(reviews)} reviews to {output}")
```

### Pattern 2: Extract Aspect-Sentiment Pairs

```python
from data.extraction import extract_semeval2014

reviews = extract_semeval2014('data/semeval2014/Restaurants_Train.xml')

pairs = []
for review in reviews:
    for aspect in review.aspect_terms:
        pairs.append({
            'text': review.text,
            'aspect': aspect.term,
            'sentiment': aspect.polarity
        })

print(f"Extracted {len(pairs)} aspect-sentiment pairs")
```

### Pattern 3: Filter by Polarity

```python
from data.extraction import extract_semeval2014

reviews = extract_semeval2014('data/semeval2014/Restaurants_Train.xml')

positive_reviews = [
    r for r in reviews
    if any(a.polarity == 'positive' for a in r.aspect_terms)
]

print(f"Found {len(positive_reviews)} reviews with positive aspects")
```

### Pattern 4: Convert to DataFrame (requires pandas)

```python
from data.extraction import extract_semeval2014
import pandas as pd

reviews = extract_semeval2014('data/semeval2014/Restaurants_Train.xml')

# Create DataFrame
data = []
for review in reviews:
    for aspect in review.aspect_terms:
        data.append({
            'review_id': review.review_id,
            'text': review.text,
            'aspect': aspect.term,
            'polarity': aspect.polarity,
            'domain': review.domain
        })

df = pd.DataFrame(data)
print(df.head())
```

## Testing

Run the test suite to verify everything works:

```bash
python test_extraction.py
```

Expected output:
```
======================================================================
Test Results: 6 passed, 0 failed
======================================================================

✓ All tests PASSED!
```

## Troubleshooting

### Problem: File not found

```python
# Make sure paths are correct
import os
path = 'data/semeval2014/Restaurants_Train.xml'
if not os.path.exists(path):
    print(f"File not found: {path}")
```

### Problem: Empty aspect lists

```python
# Filter reviews with aspects
reviews_with_aspects = [
    r for r in reviews
    if r.aspect_terms or r.opinions
]
```

### Problem: Encoding errors

The module uses UTF-8 by default. Ensure your data files are UTF-8 encoded.

## Next Steps

1. Read the full documentation: [README_EXTRACTION.md](README_EXTRACTION.md)
2. Check examples: [example_extraction.py](example_extraction.py)
3. Download SemEval datasets from official sources
4. Start extracting your data!

## File Structure

```
final_nlp/
├── data/
│   ├── extraction.py              # Main module
│   ├── semeval2014/              # Place SemEval-2014 files here
│   ├── semeval2015/              # Place SemEval-2015 files here
│   ├── semeval2016/              # Place SemEval-2016 files here
│   └── multidomain/              # Place multi-domain files here
├── output/                        # Extracted JSON files will go here
├── example_extraction.py          # Example usage
├── test_extraction.py            # Test suite
├── QUICKSTART.md                 # This file
└── README_EXTRACTION.md          # Full documentation
```

## Quick Reference: Dataset Types

| Dataset | XML Format | Supported Years | Domains |
|---------|-----------|-----------------|---------|
| SemEval-2014 | ✓ | 2014 | Laptop, Restaurant |
| SemEval-2015 | ✓ | 2015 | Restaurant, Hotel, Others |
| SemEval-2016 | ✓ | 2016 | Restaurant, Laptop, Others |
| Multi-Domain | JSON, CSV | N/A | Custom |

## Support

For issues or questions:
1. Check the full documentation
2. Review the example scripts
3. Run the test suite to identify problems
