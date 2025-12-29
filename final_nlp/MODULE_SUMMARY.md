# ABSA Data Extraction Module - Summary

## 📦 What You Got

A complete, production-ready data extraction module for ABSA (Aspect-Based Sentiment Analysis) datasets with comprehensive support for:

- ✅ **SemEval-2014 Task 4** (Laptop and Restaurant reviews)
- ✅ **SemEval-2015 Task 12** (Multi-domain sentiment analysis)
- ✅ **SemEval-2016 Task 5** (Aspect-based sentiment analysis)
- ✅ **Multi-Domain datasets** (JSON/CSV formats)

## 📁 Files Created

```
final_nlp/
├── data/
│   ├── extraction.py              # Main extraction module (800+ lines)
│   └── __init__.py               # Package initialization
├── example_extraction.py          # Comprehensive usage examples
├── test_extraction.py            # Complete test suite (6 tests)
├── QUICKSTART.md                 # Quick start guide
├── README_EXTRACTION.md          # Full documentation
├── requirements_extraction.txt   # Dependencies (standard library only)
└── MODULE_SUMMARY.md            # This file
```

## 🚀 Quick Start (30 seconds)

```python
from data.extraction import extract_semeval2014

# Extract data
reviews = extract_semeval2014('data/semeval2014/Restaurants_Train.xml')

# Use it
for review in reviews:
    print(f"Text: {review.text}")
    for aspect in review.aspect_terms:
        print(f"  {aspect.term}: {aspect.polarity}")
```

## 🎯 Key Features

### 1. Unified Interface
One consistent API for all dataset types:
```python
extractor = DatasetExtractor()
reviews = extractor.extract(file_path, dataset_type, domain)
```

### 2. Structured Data Models
Type-safe dataclasses with full IDE support:
- `Review`: Complete review with all annotations
- `AspectTerm`: Aspect with position and polarity
- `AspectCategory`: Category-level annotations
- `Opinion`: Opinion expressions with targets

### 3. Multiple Format Support
- XML (SemEval datasets)
- JSON (Multi-domain datasets)
- CSV (Custom datasets)

### 4. Built-in Analytics
```python
stats = extractor.get_statistics(reviews)
# Returns: total reviews, aspects, polarities, distributions, etc.
```

### 5. Easy Export
```python
extractor.save_to_json(reviews, 'output.json')
```

### 6. Comprehensive Testing
```bash
python test_extraction.py
# All 6 tests pass ✓
```

## 📊 Data Models Overview

### Review Object
```python
Review(
    review_id: str,
    text: str,
    aspect_terms: List[AspectTerm],
    aspect_categories: List[AspectCategory],
    opinions: List[Opinion],
    domain: str
)
```

### AspectTerm Object
```python
AspectTerm(
    term: str,              # "food"
    polarity: str,          # "positive"
    from_index: int,        # 4
    to_index: int          # 8
)
```

## 🔧 Usage Patterns

### Pattern 1: Simple Extraction
```python
from data.extraction import extract_semeval2014
reviews = extract_semeval2014('data.xml', domain='restaurant')
```

### Pattern 2: Batch Processing
```python
extractor = DatasetExtractor()
for dataset in datasets:
    reviews = extractor.extract(dataset['path'], dataset['type'], dataset['domain'])
    extractor.save_to_json(reviews, dataset['output'])
```

### Pattern 3: Custom Processing
```python
aspect_sentiment_pairs = [
    (aspect.term, aspect.polarity, review.text)
    for review in reviews
    for aspect in review.aspect_terms
]
```

## 📈 Statistics Example

```python
stats = extractor.get_statistics(reviews)
# Output:
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

## 🧪 Testing

All components are fully tested:

```bash
$ python test_extraction.py

======================================================================
ABSA Data Extraction Module - Test Suite
======================================================================

✓ Data Models Test PASSED
✓ SemEval-2014 Extraction Test PASSED
✓ SemEval-2015 Extraction Test PASSED
✓ SemEval-2016 Extraction Test PASSED
✓ Multi-Domain Extraction Test PASSED
✓ Unified Extractor Test PASSED

======================================================================
Test Results: 6 passed, 0 failed
======================================================================
```

## 📚 Documentation

1. **QUICKSTART.md** - Get started in 5 minutes
2. **README_EXTRACTION.md** - Complete documentation with examples
3. **example_extraction.py** - Working code examples
4. **test_extraction.py** - Test suite with sample data

## 🎓 Supported Dataset Formats

### SemEval-2014 XML
```xml
<sentence id="1">
  <text>The food was delicious.</text>
  <aspectTerms>
    <aspectTerm term="food" polarity="positive" from="4" to="8"/>
  </aspectTerms>
</sentence>
```

### SemEval-2015/2016 XML
```xml
<Review rid="123">
  <sentence id="123:0">
    <text>Great food.</text>
    <Opinions>
      <Opinion target="food" category="FOOD#QUALITY" polarity="positive"/>
    </Opinions>
  </sentence>
</Review>
```

### Multi-Domain JSON
```json
{
  "id": "1",
  "text": "Battery life is amazing.",
  "aspects": [
    {"term": "battery life", "polarity": "positive", "from": 0, "to": 12}
  ]
}
```

### Multi-Domain CSV
```csv
id,text,aspect,polarity
1,"Great camera",camera,positive
```

## 🔍 Advanced Features

### Custom Column Mapping (CSV)
```python
extractor.extract_from_csv(
    'data.csv',
    text_column='review_text',
    aspect_column='feature',
    polarity_column='sentiment'
)
```

### Domain Specification
```python
reviews = extract_semeval2014('data.xml', domain='laptop')
```

### Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now see detailed extraction logs
```

## 💡 Common Use Cases

### Use Case 1: Dataset Conversion
Convert SemEval XML to JSON for easier processing:
```python
extractor = DatasetExtractor()
reviews = extractor.extract('semeval.xml', 'semeval2014', 'restaurant')
extractor.save_to_json(reviews, 'semeval.json')
```

### Use Case 2: Dataset Merging
Combine multiple datasets:
```python
all_reviews = []
for dataset in ['2014', '2015', '2016']:
    reviews = extractor.extract(f'semeval{dataset}.xml', f'semeval{dataset}', 'restaurant')
    all_reviews.extend(reviews)
```

### Use Case 3: Analysis & Statistics
Analyze dataset characteristics:
```python
stats = extractor.get_statistics(reviews)
print(f"Dataset has {stats['total_reviews']} reviews")
print(f"Polarity distribution: {stats['polarity_distribution']}")
```

### Use Case 4: Data Preprocessing for ML
Extract training data:
```python
training_data = []
for review in reviews:
    for aspect in review.aspect_terms:
        training_data.append({
            'text': review.text,
            'aspect': aspect.term,
            'label': aspect.polarity,
            'position': (aspect.from_index, aspect.to_index)
        })
```

## ⚙️ Technical Details

- **Language**: Python 3.7+
- **Dependencies**: Standard library only
- **Lines of Code**: ~800 (extraction.py)
- **Test Coverage**: 6 comprehensive tests
- **Encoding**: UTF-8 throughout
- **Error Handling**: Comprehensive with informative messages
- **Type Hints**: Full typing support
- **Logging**: Integrated Python logging

## 🛠️ Maintenance & Extension

### Adding New Dataset Support

1. Create new extractor class:
```python
class NewDatasetExtractor:
    def extract_from_xml(self, xml_path):
        # Implementation
        pass
```

2. Register in DatasetExtractor:
```python
self.extractors['new_dataset'] = NewDatasetExtractor
```

### Customizing Data Models

All models are dataclasses, easy to extend:
```python
@dataclass
class Review:
    # Add new fields
    custom_field: Optional[str] = None
```

## 📝 Best Practices

1. **Always specify domain**: `extract_semeval2014(path, domain='restaurant')`
2. **Check file existence**: Use try-except for FileNotFoundError
3. **Filter empty reviews**: `[r for r in reviews if r.aspect_terms]`
4. **Use unified interface**: Prefer `DatasetExtractor` for consistency
5. **Save intermediate results**: Use `save_to_json()` for checkpoints

## 🚦 Getting Started Checklist

- [ ] Review QUICKSTART.md
- [ ] Run test suite: `python test_extraction.py`
- [ ] Try example script: `python example_extraction.py`
- [ ] Download SemEval datasets
- [ ] Extract your first dataset
- [ ] Explore the data models
- [ ] Read full documentation

## 📊 Performance

- Fast XML parsing using ElementTree
- Memory efficient (streaming for large files possible)
- Processes typical SemEval datasets in seconds
- No external dependencies for core functionality

## 🤝 Integration Examples

### With Pandas
```python
import pandas as pd
df = pd.DataFrame([r.to_dict() for r in reviews])
```

### With scikit-learn
```python
from sklearn.model_selection import train_test_split
texts = [r.text for r in reviews]
labels = [a.polarity for r in reviews for a in r.aspect_terms]
```

### With PyTorch/TensorFlow
```python
dataset = [(r.text, a.polarity) for r in reviews for a in r.aspect_terms]
```

## 🎯 Next Steps

1. **Start with QUICKSTART.md** for immediate usage
2. **Read README_EXTRACTION.md** for comprehensive guide
3. **Check example_extraction.py** for code patterns
4. **Download SemEval datasets** from official sources
5. **Begin extracting** your ABSA data!

## 📞 Support

- Documentation: See README_EXTRACTION.md
- Examples: See example_extraction.py
- Testing: Run test_extraction.py
- Issues: Check error messages and logs

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Tests**: 6/6 Passing
**Documentation**: Complete

Enjoy your ABSA data extraction! 🎉
