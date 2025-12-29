"""
Script simple pour extraire les données de restaurant_trial.xml
"""

import sys
import codecs

# Fix encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from data.extraction import extract_semeval2014, DatasetExtractor

print("="*60)
print("Extraction des données - Restaurant Trial")
print("="*60)

# Extraction des reviews
print("\n1. Extraction des données...")
reviews = extract_semeval2014(
    'data/raw/restaurant_trials.xml',
    domain='restaurant'
)

print(f"✓ {len(reviews)} reviews extraites!\n")

# Afficher quelques statistiques
extractor = DatasetExtractor()
stats = extractor.get_statistics(reviews)

print("2. Statistiques:")
print(f"   - Total reviews: {stats['total_reviews']}")
print(f"   - Total aspect terms: {stats['total_aspect_terms']}")
print(f"   - Total aspect categories: {stats['total_aspect_categories']}")
print(f"   - Distribution des polarités:")
for polarity, count in stats['polarity_distribution'].items():
    print(f"     • {polarity}: {count}")

# Afficher quelques exemples
print("\n3. Exemples de reviews:\n")
for i, review in enumerate(reviews[:3], 1):
    print(f"Review #{i} (ID: {review.review_id}):")
    print(f"   Text: {review.text}")
    print(f"   Aspects:")
    for aspect in review.aspect_terms:
        print(f"      → {aspect.term}: {aspect.polarity}")
    print()

# Sauvegarder en JSON
output_file = 'output/restaurant_trial_extracted.json'
print(f"4. Sauvegarde en JSON: {output_file}")
extractor.save_to_json(reviews, output_file)
print("✓ Sauvegardé!\n")

print("="*60)
print("Extraction terminée!")
print("="*60)
print(f"\nTu peux maintenant utiliser le fichier JSON:")
print(f"  → {output_file}")
