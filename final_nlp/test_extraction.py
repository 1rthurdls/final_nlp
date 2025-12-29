"""
Test script for the ABSA Data Extraction Module

This script tests the extraction module with sample data or mock data.
"""

import os
import sys
import tempfile

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
from data.extraction import (
    DatasetExtractor,
    SemEval2014Extractor,
    SemEval2015Extractor,
    SemEval2016Extractor,
    MultiDomainExtractor,
    Review,
    AspectTerm,
    AspectCategory,
    Opinion
)
import json


def create_sample_semeval2014_xml():
    """Create a sample SemEval-2014 XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="1">
    <text>The food was delicious but the service was slow.</text>
    <aspectTerms>
      <aspectTerm term="food" polarity="positive" from="4" to="8"/>
      <aspectTerm term="service" polarity="negative" from="35" to="42"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="FOOD#QUALITY" polarity="positive"/>
      <aspectCategory category="SERVICE#GENERAL" polarity="negative"/>
    </aspectCategories>
  </sentence>
  <sentence id="2">
    <text>Great atmosphere and reasonable prices.</text>
    <aspectTerms>
      <aspectTerm term="atmosphere" polarity="positive" from="6" to="16"/>
      <aspectTerm term="prices" polarity="positive" from="32" to="38"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="AMBIENCE#GENERAL" polarity="positive"/>
      <aspectCategory category="PRICE#GENERAL" polarity="positive"/>
    </aspectCategories>
  </sentence>
</sentences>
"""
    return xml_content


def create_sample_semeval2015_xml():
    """Create a sample SemEval-2015 XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
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
  <Review rid="124">
    <sentences>
      <sentence id="124:0">
        <text>The staff was friendly.</text>
        <Opinions>
          <Opinion target="staff" category="SERVICE#GENERAL" polarity="positive" from="4" to="9"/>
        </Opinions>
      </sentence>
    </sentences>
  </Review>
</Reviews>
"""
    return xml_content


def create_sample_semeval2016_xml():
    """Create a sample SemEval-2016 XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Reviews>
  <Review rid="456">
    <sentences>
      <sentence id="456:0">
        <text>Amazing pizza but noisy environment.</text>
        <Opinions>
          <Opinion target="pizza" category="FOOD#QUALITY" polarity="positive" from="8" to="13"/>
          <Opinion target="environment" category="AMBIENCE#GENERAL" polarity="negative" from="24" to="35"/>
        </Opinions>
      </sentence>
    </sentences>
  </Review>
</Reviews>
"""
    return xml_content


def create_sample_multidomain_json():
    """Create a sample Multi-Domain JSON file for testing."""
    data = [
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
        },
        {
            "id": "review_2",
            "text": "Screen quality is poor but performance is great.",
            "domain": "electronics",
            "aspects": [
                {
                    "term": "screen quality",
                    "polarity": "negative",
                    "from": 0,
                    "to": 14
                },
                {
                    "term": "performance",
                    "polarity": "positive",
                    "from": 27,
                    "to": 38
                }
            ]
        }
    ]
    return json.dumps(data, indent=2)


def test_semeval2014_extraction():
    """Test SemEval-2014 extraction."""
    print("\n" + "="*60)
    print("Testing SemEval-2014 Extraction")
    print("="*60)

    # Create temporary XML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
        f.write(create_sample_semeval2014_xml())
        temp_file = f.name

    try:
        # Test extraction
        extractor = SemEval2014Extractor(domain='restaurant')
        reviews = extractor.extract_from_xml(temp_file)

        assert len(reviews) == 2, f"Expected 2 reviews, got {len(reviews)}"
        print(f"✓ Extracted {len(reviews)} reviews")

        # Test first review
        review1 = reviews[0]
        assert review1.review_id == '1', f"Expected ID '1', got '{review1.review_id}'"
        assert len(review1.aspect_terms) == 2, f"Expected 2 aspect terms, got {len(review1.aspect_terms)}"
        assert len(review1.aspect_categories) == 2, f"Expected 2 categories, got {len(review1.aspect_categories)}"
        print(f"✓ Review 1: {len(review1.aspect_terms)} aspect terms, {len(review1.aspect_categories)} categories")

        # Test aspect term details
        aspect1 = review1.aspect_terms[0]
        assert aspect1.term == 'food', f"Expected 'food', got '{aspect1.term}'"
        assert aspect1.polarity == 'positive', f"Expected 'positive', got '{aspect1.polarity}'"
        print(f"✓ Aspect term: '{aspect1.term}' with polarity '{aspect1.polarity}'")

        print("✓ SemEval-2014 extraction test PASSED")

    finally:
        os.unlink(temp_file)


def test_semeval2015_extraction():
    """Test SemEval-2015 extraction."""
    print("\n" + "="*60)
    print("Testing SemEval-2015 Extraction")
    print("="*60)

    # Create temporary XML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
        f.write(create_sample_semeval2015_xml())
        temp_file = f.name

    try:
        # Test extraction
        extractor = SemEval2015Extractor(domain='restaurant')
        reviews = extractor.extract_from_xml(temp_file)

        assert len(reviews) == 2, f"Expected 2 reviews, got {len(reviews)}"
        print(f"✓ Extracted {len(reviews)} reviews")

        # Test first review
        review1 = reviews[0]
        assert len(review1.opinions) == 2, f"Expected 2 opinions, got {len(review1.opinions)}"
        print(f"✓ Review 1: {len(review1.opinions)} opinions")

        # Test opinion details
        opinion1 = review1.opinions[0]
        assert opinion1.target == 'food', f"Expected 'food', got '{opinion1.target}'"
        assert opinion1.category == 'FOOD#QUALITY', f"Expected 'FOOD#QUALITY', got '{opinion1.category}'"
        assert opinion1.polarity == 'positive', f"Expected 'positive', got '{opinion1.polarity}'"
        print(f"✓ Opinion: target='{opinion1.target}', category='{opinion1.category}', polarity='{opinion1.polarity}'")

        print("✓ SemEval-2015 extraction test PASSED")

    finally:
        os.unlink(temp_file)


def test_semeval2016_extraction():
    """Test SemEval-2016 extraction."""
    print("\n" + "="*60)
    print("Testing SemEval-2016 Extraction")
    print("="*60)

    # Create temporary XML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
        f.write(create_sample_semeval2016_xml())
        temp_file = f.name

    try:
        # Test extraction
        extractor = SemEval2016Extractor(domain='restaurant')
        reviews = extractor.extract_from_xml(temp_file)

        assert len(reviews) == 1, f"Expected 1 review, got {len(reviews)}"
        print(f"✓ Extracted {len(reviews)} review")

        # Test review
        review = reviews[0]
        assert len(review.opinions) == 2, f"Expected 2 opinions, got {len(review.opinions)}"
        print(f"✓ Review: {len(review.opinions)} opinions")

        print("✓ SemEval-2016 extraction test PASSED")

    finally:
        os.unlink(temp_file)


def test_multidomain_extraction():
    """Test Multi-Domain extraction."""
    print("\n" + "="*60)
    print("Testing Multi-Domain Extraction")
    print("="*60)

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write(create_sample_multidomain_json())
        temp_file = f.name

    try:
        # Test extraction
        extractor = MultiDomainExtractor()
        reviews = extractor.extract_from_json(temp_file, domain='electronics')

        assert len(reviews) == 2, f"Expected 2 reviews, got {len(reviews)}"
        print(f"✓ Extracted {len(reviews)} reviews")

        # Test first review
        review1 = reviews[0]
        assert review1.domain == 'electronics', f"Expected 'electronics', got '{review1.domain}'"
        assert len(review1.aspect_terms) == 1, f"Expected 1 aspect term, got {len(review1.aspect_terms)}"
        print(f"✓ Review 1: domain='{review1.domain}', {len(review1.aspect_terms)} aspect terms")

        # Test second review
        review2 = reviews[1]
        assert len(review2.aspect_terms) == 2, f"Expected 2 aspect terms, got {len(review2.aspect_terms)}"
        print(f"✓ Review 2: {len(review2.aspect_terms)} aspect terms")

        print("✓ Multi-Domain extraction test PASSED")

    finally:
        os.unlink(temp_file)


def test_unified_extractor():
    """Test unified DatasetExtractor."""
    print("\n" + "="*60)
    print("Testing Unified DatasetExtractor")
    print("="*60)

    # Create temporary XML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
        f.write(create_sample_semeval2014_xml())
        temp_file = f.name

    try:
        # Test unified extractor
        extractor = DatasetExtractor()
        reviews = extractor.extract(temp_file, dataset_type='semeval2014', domain='restaurant')

        assert len(reviews) == 2, f"Expected 2 reviews, got {len(reviews)}"
        print(f"✓ Extracted {len(reviews)} reviews using unified extractor")

        # Test statistics
        stats = extractor.get_statistics(reviews)
        assert stats['total_reviews'] == 2, f"Expected 2 total reviews, got {stats['total_reviews']}"
        assert stats['total_aspect_terms'] == 4, f"Expected 4 aspect terms, got {stats['total_aspect_terms']}"
        print(f"✓ Statistics: {stats['total_reviews']} reviews, {stats['total_aspect_terms']} aspect terms")

        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as out_f:
            output_file = out_f.name

        extractor.save_to_json(reviews, output_file)
        assert os.path.exists(output_file), "Output JSON file not created"

        # Verify JSON content
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert len(saved_data) == 2, f"Expected 2 reviews in JSON, got {len(saved_data)}"
            print(f"✓ Saved {len(saved_data)} reviews to JSON")

        os.unlink(output_file)
        print("✓ Unified extractor test PASSED")

    finally:
        os.unlink(temp_file)


def test_data_models():
    """Test data models."""
    print("\n" + "="*60)
    print("Testing Data Models")
    print("="*60)

    # Test AspectTerm
    aspect = AspectTerm(term="food", polarity="positive", from_index=0, to_index=4)
    assert aspect.term == "food"
    assert aspect.polarity == "positive"
    aspect_dict = aspect.to_dict()
    assert aspect_dict['term'] == "food"
    print("✓ AspectTerm model works correctly")

    # Test AspectCategory
    category = AspectCategory(category="FOOD#QUALITY", polarity="positive")
    assert category.category == "FOOD#QUALITY"
    category_dict = category.to_dict()
    assert category_dict['category'] == "FOOD#QUALITY"
    print("✓ AspectCategory model works correctly")

    # Test Opinion
    opinion = Opinion(target="food", category="FOOD#QUALITY", polarity="positive")
    assert opinion.target == "food"
    opinion_dict = opinion.to_dict()
    assert 'target' in opinion_dict
    print("✓ Opinion model works correctly")

    # Test Review
    review = Review(
        review_id="1",
        text="Test review",
        aspect_terms=[aspect],
        aspect_categories=[category],
        opinions=[opinion],
        domain="restaurant"
    )
    assert review.review_id == "1"
    review_dict = review.to_dict()
    assert review_dict['review_id'] == "1"
    assert len(review_dict['aspect_terms']) == 1
    print("✓ Review model works correctly")

    print("✓ Data models test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("ABSA Data Extraction Module - Test Suite")
    print("="*70)

    tests = [
        ("Data Models", test_data_models),
        ("SemEval-2014 Extraction", test_semeval2014_extraction),
        ("SemEval-2015 Extraction", test_semeval2015_extraction),
        ("SemEval-2016 Extraction", test_semeval2016_extraction),
        ("Multi-Domain Extraction", test_multidomain_extraction),
        ("Unified Extractor", test_unified_extractor),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {str(e)}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("\n✓ All tests PASSED!")
    else:
        print(f"\n✗ {failed} test(s) FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
