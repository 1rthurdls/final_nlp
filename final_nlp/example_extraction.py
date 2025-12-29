"""
Example usage of the ABSA Data Extraction Module

This script demonstrates how to use the extraction module with different datasets.
"""

from data.extraction import (
    DatasetExtractor,
    extract_semeval2014,
    extract_semeval2015,
    extract_semeval2016,
    extract_multidomain
)
import json


def example_semeval2014():
    """Example: Extract SemEval-2014 data."""
    print("\n" + "="*60)
    print("SemEval-2014 Task 4 - Restaurant & Laptop Reviews")
    print("="*60)

    # Method 1: Using convenience function
    try:
        reviews_restaurant = extract_semeval2014(
            'data/semeval2014/Restaurants_Train.xml',
            domain='restaurant'
        )
        print(f"✓ Extracted {len(reviews_restaurant)} restaurant reviews")

        # Show sample review
        if reviews_restaurant:
            sample = reviews_restaurant[0]
            print(f"\nSample Review:")
            print(f"  ID: {sample.review_id}")
            print(f"  Text: {sample.text[:100]}...")
            print(f"  Aspect Terms: {len(sample.aspect_terms)}")
            print(f"  Aspect Categories: {len(sample.aspect_categories)}")

    except FileNotFoundError:
        print("✗ Restaurant data file not found")

    try:
        reviews_laptop = extract_semeval2014(
            'data/semeval2014/Laptop_Train.xml',
            domain='laptop'
        )
        print(f"✓ Extracted {len(reviews_laptop)} laptop reviews")

    except FileNotFoundError:
        print("✗ Laptop data file not found")


def example_semeval2015():
    """Example: Extract SemEval-2015 data."""
    print("\n" + "="*60)
    print("SemEval-2015 Task 12")
    print("="*60)

    try:
        reviews = extract_semeval2015(
            'data/semeval2015/ABSA15_RestaurantsTrain.xml',
            domain='restaurant'
        )
        print(f"✓ Extracted {len(reviews)} reviews")

        # Show sample review with opinions
        if reviews:
            sample = reviews[0]
            print(f"\nSample Review:")
            print(f"  ID: {sample.review_id}")
            print(f"  Text: {sample.text[:100]}...")
            print(f"  Opinions: {len(sample.opinions)}")
            if sample.opinions:
                op = sample.opinions[0]
                print(f"    - Target: {op.target}, Category: {op.category}, Polarity: {op.polarity}")

    except FileNotFoundError:
        print("✗ SemEval-2015 data file not found")


def example_semeval2016():
    """Example: Extract SemEval-2016 data."""
    print("\n" + "="*60)
    print("SemEval-2016 Task 5")
    print("="*60)

    try:
        reviews = extract_semeval2016(
            'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
            domain='restaurant'
        )
        print(f"✓ Extracted {len(reviews)} reviews")

        # Show sample review
        if reviews:
            sample = reviews[0]
            print(f"\nSample Review:")
            print(f"  ID: {sample.review_id}")
            print(f"  Text: {sample.text[:100]}...")
            print(f"  Opinions: {len(sample.opinions)}")

    except FileNotFoundError:
        print("✗ SemEval-2016 data file not found")


def example_multidomain():
    """Example: Extract Multi-Domain data."""
    print("\n" + "="*60)
    print("Multi-Domain Aspect Extraction")
    print("="*60)

    # Example with JSON
    try:
        reviews_json = extract_multidomain(
            'data/multidomain/electronics.json',
            domain='electronics'
        )
        print(f"✓ Extracted {len(reviews_json)} reviews from JSON")

    except FileNotFoundError:
        print("✗ Multi-domain JSON file not found")

    # Example with CSV
    try:
        extractor = DatasetExtractor()
        reviews_csv = extractor.extract(
            'data/multidomain/products.csv',
            dataset_type='multidomain',
            domain='products',
            text_column='review_text',
            aspect_column='aspect',
            polarity_column='sentiment'
        )
        print(f"✓ Extracted {len(reviews_csv)} reviews from CSV")

    except FileNotFoundError:
        print("✗ Multi-domain CSV file not found")


def example_unified_extraction():
    """Example: Using the unified DatasetExtractor."""
    print("\n" + "="*60)
    print("Unified DatasetExtractor Example")
    print("="*60)

    extractor = DatasetExtractor()

    # Define datasets to process
    datasets = [
        {
            'path': 'data/semeval2014/Restaurants_Train.xml',
            'type': 'semeval2014',
            'domain': 'restaurant',
            'output': 'output/semeval2014_restaurant.json'
        },
        {
            'path': 'data/semeval2015/ABSA15_RestaurantsTrain.xml',
            'type': 'semeval2015',
            'domain': 'restaurant',
            'output': 'output/semeval2015_restaurant.json'
        },
        {
            'path': 'data/semeval2016/ABSA16_Restaurants_Train_SB1_v2.xml',
            'type': 'semeval2016',
            'domain': 'restaurant',
            'output': 'output/semeval2016_restaurant.json'
        }
    ]

    all_reviews = []

    for dataset in datasets:
        try:
            print(f"\nProcessing {dataset['type']} - {dataset['domain']}...")
            reviews = extractor.extract(
                dataset['path'],
                dataset_type=dataset['type'],
                domain=dataset['domain']
            )

            # Get statistics
            stats = extractor.get_statistics(reviews)
            print(f"  Total reviews: {stats['total_reviews']}")
            print(f"  Total aspect terms: {stats['total_aspect_terms']}")
            print(f"  Total opinions: {stats['total_opinions']}")
            print(f"  Polarity distribution: {stats['polarity_distribution']}")

            # Save to JSON
            extractor.save_to_json(reviews, dataset['output'])
            print(f"  ✓ Saved to {dataset['output']}")

            all_reviews.extend(reviews)

        except FileNotFoundError:
            print(f"  ✗ File not found: {dataset['path']}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    # Combined statistics
    if all_reviews:
        print("\n" + "-"*60)
        print("Combined Statistics")
        print("-"*60)
        combined_stats = extractor.get_statistics(all_reviews)
        print(json.dumps(combined_stats, indent=2))


def example_custom_processing():
    """Example: Custom processing of extracted data."""
    print("\n" + "="*60)
    print("Custom Data Processing Example")
    print("="*60)

    try:
        reviews = extract_semeval2014(
            'data/semeval2014/Restaurants_Train.xml',
            domain='restaurant'
        )

        # Extract all aspect terms
        all_aspects = []
        for review in reviews:
            for aspect_term in review.aspect_terms:
                all_aspects.append({
                    'text': review.text,
                    'aspect': aspect_term.term,
                    'polarity': aspect_term.polarity,
                    'position': (aspect_term.from_index, aspect_term.to_index)
                })

        print(f"✓ Extracted {len(all_aspects)} aspect-polarity pairs")

        # Count unique aspects
        unique_aspects = set(a['aspect'].lower() for a in all_aspects)
        print(f"✓ Found {len(unique_aspects)} unique aspects")

        # Polarity distribution
        polarity_dist = {}
        for aspect in all_aspects:
            pol = aspect['polarity']
            polarity_dist[pol] = polarity_dist.get(pol, 0) + 1

        print("\nPolarity Distribution:")
        for polarity, count in sorted(polarity_dist.items()):
            print(f"  {polarity}: {count} ({count/len(all_aspects)*100:.1f}%)")

    except FileNotFoundError:
        print("✗ Data file not found")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ABSA Data Extraction Module - Examples")
    print("="*60)

    # Run examples
    example_semeval2014()
    example_semeval2015()
    example_semeval2016()
    example_multidomain()
    example_unified_extraction()
    example_custom_processing()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
