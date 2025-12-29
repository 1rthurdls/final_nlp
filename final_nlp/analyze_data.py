"""
Analyse simple des données extraites
"""

import sys
import codecs
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from data.extraction import extract_semeval2014
from collections import Counter

# Charger les données
reviews = extract_semeval2014('data/raw/restaurant_trials.xml')

print("="*60)
print("ANALYSE DES DONNÉES - RESTAURANT REVIEWS")
print("="*60)

# 1. Compter les aspects les plus fréquents
print("\n1. TOP 10 des aspects les plus mentionnés:")
all_aspects = [aspect.term.lower() for r in reviews for aspect in r.aspect_terms]
aspect_counts = Counter(all_aspects)
for aspect, count in aspect_counts.most_common(10):
    print(f"   {count:3d}x - {aspect}")

# 2. Aspects positifs vs négatifs
print("\n2. Sentiments par aspect:")
aspect_sentiments = {}
for review in reviews:
    for aspect in review.aspect_terms:
        term = aspect.term.lower()
        if term not in aspect_sentiments:
            aspect_sentiments[term] = {'positive': 0, 'negative': 0, 'neutral': 0}
        aspect_sentiments[term][aspect.polarity] += 1

# Afficher les 5 aspects les plus controversés
print("\n   Aspects avec opinions mixtes:")
controversial = []
for aspect, sentiments in aspect_sentiments.items():
    if sentiments['positive'] > 0 and sentiments['negative'] > 0:
        total = sum(sentiments.values())
        controversial.append((aspect, sentiments, total))

controversial.sort(key=lambda x: x[2], reverse=True)
for aspect, sentiments, total in controversial[:5]:
    print(f"   • {aspect}:")
    print(f"     Positif: {sentiments['positive']}, Négatif: {sentiments['negative']}, Neutre: {sentiments['neutral']}")

# 3. Catégories d'aspects
print("\n3. Catégories d'aspects:")
categories = [cat.category for r in reviews for cat in r.aspect_categories]
cat_counts = Counter(categories)
for cat, count in cat_counts.most_common():
    print(f"   {count:3d}x - {cat}")

# 4. Reviews sans aspects
reviews_without_aspects = [r for r in reviews if not r.aspect_terms]
print(f"\n4. Reviews sans aspects explicites: {len(reviews_without_aspects)}/{len(reviews)}")

# 5. Longueur moyenne du texte
avg_length = sum(len(r.text) for r in reviews) / len(reviews)
print(f"\n5. Longueur moyenne des reviews: {avg_length:.1f} caractères")

# 6. Reviews les plus positives
print("\n6. TOP 3 reviews les plus positives:")
positive_reviews = []
for review in reviews:
    pos_count = sum(1 for a in review.aspect_terms if a.polarity == 'positive')
    if pos_count > 0:
        positive_reviews.append((review, pos_count))

positive_reviews.sort(key=lambda x: x[1], reverse=True)
for i, (review, count) in enumerate(positive_reviews[:3], 1):
    print(f"\n   #{i} ({count} aspects positifs):")
    print(f"   \"{review.text[:80]}...\"")
    aspects = [a.term for a in review.aspect_terms if a.polarity == 'positive']
    print(f"   Aspects: {', '.join(aspects)}")

print("\n" + "="*60)
