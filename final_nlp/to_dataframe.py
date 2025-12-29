"""
Convertir les données en DataFrame Pandas pour analyse ML
"""

import sys
import codecs
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from data.extraction import extract_semeval2014

try:
    import pandas as pd

    # Charger les données
    reviews = extract_semeval2014('data/raw/restaurant_trials.xml')

    # Créer un DataFrame avec une ligne par aspect
    data = []
    for review in reviews:
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

    print("DataFrame créé!")
    print(f"\nShape: {df.shape}")
    print(f"\nPremières lignes:")
    print(df.head(10))

    print(f"\nDistribution des polarités:")
    print(df['polarity'].value_counts())

    print(f"\nAspects les plus fréquents:")
    print(df['aspect'].value_counts().head(10))

    # Sauvegarder en CSV
    csv_file = 'output/restaurant_aspects.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n✓ Sauvegardé en CSV: {csv_file}")

    # Sauvegarder en Excel (optionnel)
    try:
        excel_file = 'output/restaurant_aspects.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"✓ Sauvegardé en Excel: {excel_file}")
    except:
        print("  (Excel non disponible - installe openpyxl si besoin)")

except ImportError:
    print("Pandas n'est pas installé!")
    print("Installe-le avec: pip install pandas")
