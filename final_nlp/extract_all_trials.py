"""
Extraire TOUS les fichiers trial (restaurant + laptop)
"""

import sys
import codecs
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from data.extraction import extract_semeval2014, DatasetExtractor
import pandas as pd

print("="*60)
print("EXTRACTION COMPLÈTE - TOUS LES DATASETS TRIAL")
print("="*60)

extractor = DatasetExtractor()
all_reviews = []

# Datasets à extraire
datasets = [
    {
        'path': 'data/raw/restaurant_trials.xml',
        'domain': 'restaurant',
        'name': 'Restaurant'
    },
    {
        'path': 'data/raw/laptop_trials.xml',
        'domain': 'laptop',
        'name': 'Laptop'
    }
]

# Extraire chaque dataset
for dataset in datasets:
    try:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*60}")

        reviews = extract_semeval2014(dataset['path'], domain=dataset['domain'])

        # Stats
        stats = extractor.get_statistics(reviews)
        print(f"✓ Reviews: {stats['total_reviews']}")
        print(f"✓ Aspect terms: {stats['total_aspect_terms']}")
        print(f"✓ Polarités: {stats['polarity_distribution']}")

        all_reviews.extend(reviews)

    except FileNotFoundError:
        print(f"✗ Fichier non trouvé: {dataset['path']}")
    except Exception as e:
        print(f"✗ Erreur: {e}")

# Stats totales
print(f"\n{'='*60}")
print("STATISTIQUES TOTALES")
print(f"{'='*60}")

total_stats = extractor.get_statistics(all_reviews)
print(f"\nTotal reviews: {total_stats['total_reviews']}")
print(f"Total aspect terms: {total_stats['total_aspect_terms']}")
print(f"Total aspect categories: {total_stats['total_aspect_categories']}")
print(f"Domaines: {', '.join(total_stats['domains'])}")
print(f"\nDistribution des polarités:")
for pol, count in total_stats['polarity_distribution'].items():
    pct = (count / sum(total_stats['polarity_distribution'].values())) * 100
    print(f"  • {pol}: {count} ({pct:.1f}%)")

# Sauvegarder JSON combiné
json_output = 'output/all_trials_combined.json'
extractor.save_to_json(all_reviews, json_output)
print(f"\n✓ JSON sauvegardé: {json_output}")

# Créer DataFrame combiné
print("\n" + "="*60)
print("CRÉATION DU DATAFRAME COMBINÉ")
print("="*60)

data = []
for review in all_reviews:
    for aspect in review.aspect_terms:
        data.append({
            'review_id': review.review_id,
            'text': review.text,
            'aspect': aspect.term,
            'polarity': aspect.polarity,
            'from': aspect.from_index,
            'to': aspect.to_index,
            'domain': review.domain
        })

df = pd.DataFrame(data)
print(f"\nDataFrame shape: {df.shape}")
print(f"Total lignes (aspects): {len(df)}")

# Analyser par domaine
print("\nRépartition par domaine:")
print(df['domain'].value_counts())

print("\nPolarités par domaine:")
print(pd.crosstab(df['domain'], df['polarity']))

# Sauvegarder CSV
csv_output = 'output/all_trials_combined.csv'
df.to_csv(csv_output, index=False, encoding='utf-8')
print(f"\n✓ CSV sauvegardé: {csv_output}")

# Sauvegarder Excel
try:
    excel_output = 'output/all_trials_combined.xlsx'
    df.to_excel(excel_output, index=False)
    print(f"✓ Excel sauvegardé: {excel_output}")
except:
    pass

print("\n" + "="*60)
print("EXTRACTION TERMINÉE!")
print("="*60)
print(f"\nFichiers créés:")
print(f"  1. {json_output}")
print(f"  2. {csv_output}")
print(f"  3. {excel_output} (si disponible)")
