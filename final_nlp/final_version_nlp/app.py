"""
Streamlit web application for ABSA demo.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoConfig
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import yaml
import os
import time

from models.advanced import BertForABSA
from models.baseline import RuleBasedABSA
from src.preprocessing import SemEvalDataLoader


# Page configuration
st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path: str, config_path: str = "configs/config.yaml"):
    """Load trained model."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load model
    bert_config = AutoConfig.from_pretrained(config['model']['name'])
    model = BertForABSA(
        bert_config,
        num_aspect_labels=config['model']['num_aspect_labels'],
        num_sentiment_labels=config['model']['num_aspect_labels']
    )

    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model, tokenizer, config


@st.cache_resource
def load_baseline_model():
    """Load rule-based baseline."""
    return RuleBasedABSA()


def predict_aspects_sentiments(
    model,
    tokenizer,
    text: str,
    device: str = 'cpu'
) -> Dict:
    """
    Predict aspects and sentiments.

    Args:
        model: ABSA model
        tokenizer: Tokenizer
        text: Input text
        device: Device to use

    Returns:
        Dictionary with predictions
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Get predictions
    aspect_logits = outputs['aspect_logits']
    sentiment_logits = outputs['sentiment_logits']

    aspect_preds = torch.argmax(aspect_logits, dim=-1)[0]
    sentiment_preds = torch.argmax(sentiment_logits, dim=-1)[0]

    # Convert tokens back to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Extract aspects
    aspects = []
    current_aspect = []
    current_sentiment = None
    aspect_start = -1

    label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    for i, (token, aspect_label, sentiment_label) in enumerate(
        zip(tokens, aspect_preds, sentiment_preds)
    ):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        label_name = label_map.get(aspect_label.item(), 'O')

        if label_name == 'B-ASPECT':
            # Save previous aspect
            if current_aspect:
                aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
                aspects.append({
                    'aspect': aspect_text,
                    'sentiment': sentiment_map.get(current_sentiment, 'neutral'),
                    'start': aspect_start,
                    'end': i
                })

            # Start new aspect
            current_aspect = [token]
            current_sentiment = sentiment_label.item()
            aspect_start = i

        elif label_name == 'I-ASPECT' and current_aspect:
            current_aspect.append(token)
            # Update sentiment if different
            if sentiment_label.item() != -100:
                current_sentiment = sentiment_label.item()

        else:
            # Save previous aspect
            if current_aspect:
                aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
                aspects.append({
                    'aspect': aspect_text,
                    'sentiment': sentiment_map.get(current_sentiment, 'neutral'),
                    'start': aspect_start,
                    'end': i
                })
                current_aspect = []
                current_sentiment = None

    # Save last aspect
    if current_aspect:
        aspect_text = tokenizer.convert_tokens_to_string(current_aspect)
        aspects.append({
            'aspect': aspect_text,
            'sentiment': sentiment_map.get(current_sentiment, 'neutral'),
            'start': aspect_start,
            'end': len(tokens)
        })

    return {
        'text': text,
        'aspects': aspects,
        'tokens': tokens
    }


def highlight_text(text: str, aspects: List[Dict]) -> str:
    """
    Highlight aspects in text with HTML.

    Args:
        text: Input text
        aspects: List of aspect dictionaries

    Returns:
        HTML string with highlighted text
    """
    # Sort aspects by start position (reverse to avoid index issues)
    sorted_aspects = sorted(aspects, key=lambda x: x.get('start', 0), reverse=True)

    highlighted = text

    # Color mapping
    color_map = {
        'positive': '#90EE90',  # Light green
        'negative': '#FFB6C1',  # Light red
        'neutral': '#FFD700'    # Gold
    }

    for aspect in sorted_aspects:
        aspect_term = aspect['aspect'].replace('##', '')  # Remove BERT artifacts
        sentiment = aspect['sentiment']
        color = color_map.get(sentiment, '#FFFFFF')

        # Find aspect in text (case-insensitive)
        import re
        pattern = re.compile(re.escape(aspect_term), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px; font-weight: bold;">{aspect_term}</span>',
            highlighted,
            count=1
        )

    return highlighted


def create_sentiment_chart(aspects: List[Dict]) -> go.Figure:
    """Create sentiment distribution chart."""
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    for aspect in aspects:
        sentiment = aspect['sentiment']
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    df = pd.DataFrame({
        'Sentiment': list(sentiment_counts.keys()),
        'Count': list(sentiment_counts.values())
    })

    colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
    df['Color'] = df['Sentiment'].map(colors)

    fig = go.Figure(data=[
        go.Bar(
            x=df['Sentiment'],
            y=df['Count'],
            marker_color=df['Color'],
            text=df['Count'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=300,
        showlegend=False
    )

    return fig


def main():
    """Main Streamlit app."""
    # Title and description
    st.title("🎯 Aspect-Based Sentiment Analysis")
    st.markdown("""
    This application performs aspect-based sentiment analysis on restaurant reviews.
    It extracts specific aspects (e.g., food, service, ambience) and determines the sentiment
    (positive, negative, neutral) for each aspect.
    """)

    # Sidebar
    st.sidebar.header("Configuration")

    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["BERT-based (Advanced)", "Rule-based (Baseline)"]
    )

    # Load model
    if model_type == "BERT-based (Advanced)":
        model_path = "models/checkpoints/best_model.pt"

        if not os.path.exists(model_path):
            st.sidebar.warning("⚠️ Trained model not found. Please train the model first.")
            model, tokenizer, config = None, None, None
        else:
            with st.spinner("Loading model..."):
                model, tokenizer, config = load_model(model_path)
            st.sidebar.success("✅ Model loaded successfully!")
    else:
        model = load_baseline_model()
        tokenizer, config = None, None
        st.sidebar.success("✅ Baseline model loaded!")

    # Examples
    st.sidebar.header("Example Reviews")
    examples = [
        "The food was amazing but the service was terrible.",
        "Great ambience and reasonable prices, highly recommended!",
        "The pizza was delicious and the staff were very friendly.",
        "Worst experience ever. The food was cold and the waiter was rude.",
        "Nice restaurant with good atmosphere, but the portions are too small."
    ]

    example_choice = st.sidebar.selectbox(
        "Choose an example",
        ["Custom input"] + examples
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Input")

        if example_choice == "Custom input":
            text_input = st.text_area(
                "Enter your restaurant review:",
                height=150,
                placeholder="E.g., The food was delicious but the service was slow..."
            )
        else:
            text_input = st.text_area(
                "Enter your restaurant review:",
                value=example_choice,
                height=150
            )

        # Analyze button
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            elif model is None and model_type == "BERT-based (Advanced)":
                st.error("❌ Model not loaded. Please train the model first.")
            else:
                # Prediction
                start_time = time.time()

                with st.spinner("Analyzing..."):
                    if model_type == "BERT-based (Advanced)":
                        predictions = predict_aspects_sentiments(
                            model, tokenizer, text_input
                        )
                        aspects = predictions['aspects']
                    else:
                        # Rule-based prediction
                        result = model.predict(text_input)
                        aspects = [
                            {
                                'aspect': p['aspect'],
                                'sentiment': p['sentiment'],
                                'start': p['start'],
                                'end': p['end']
                            }
                            for p in result['predictions']
                        ]

                inference_time = time.time() - start_time

                # Display results
                st.header("Results")

                if not aspects:
                    st.info("ℹ️ No aspects detected in the text.")
                else:
                    # Highlighted text
                    st.subheader("Highlighted Text")
                    highlighted = highlight_text(text_input, aspects)
                    st.markdown(highlighted, unsafe_allow_html=True)

                    # Legend
                    st.markdown("""
                    <div style="margin-top: 10px;">
                        <span style="background-color: #90EE90; padding: 2px 5px; border-radius: 3px; margin-right: 10px;">Positive</span>
                        <span style="background-color: #FFB6C1; padding: 2px 5px; border-radius: 3px; margin-right: 10px;">Negative</span>
                        <span style="background-color: #FFD700; padding: 2px 5px; border-radius: 3px;">Neutral</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Aspect table
                    st.subheader("Extracted Aspects")
                    aspect_df = pd.DataFrame(aspects)

                    # Format table
                    if 'start' in aspect_df.columns:
                        aspect_df = aspect_df[['aspect', 'sentiment']]

                    st.dataframe(
                        aspect_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Sentiment chart
                    st.subheader("Sentiment Distribution")
                    fig = create_sentiment_chart(aspects)
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics
                    st.subheader("Metrics")
                    metric_cols = st.columns(3)

                    with metric_cols[0]:
                        st.metric("Total Aspects", len(aspects))

                    with metric_cols[1]:
                        positive_count = sum(1 for a in aspects if a['sentiment'] == 'positive')
                        st.metric("Positive", positive_count)

                    with metric_cols[2]:
                        negative_count = sum(1 for a in aspects if a['sentiment'] == 'negative')
                        st.metric("Negative", negative_count)

                    # Performance
                    st.caption(f"⏱️ Inference time: {inference_time*1000:.2f} ms")

    with col2:
        st.header("Information")

        st.markdown("""
        ### How it works

        1. **Input**: Enter a restaurant review
        2. **Aspect Extraction**: The model identifies specific aspects (food, service, etc.)
        3. **Sentiment Classification**: Each aspect is classified as positive, negative, or neutral
        4. **Visualization**: Results are highlighted and visualized

        ### Sentiment Colors

        - 🟢 **Positive**: Green
        - 🔴 **Negative**: Red
        - 🟡 **Neutral**: Yellow

        ### Model Information

        **BERT-based Model**:
        - Joint learning architecture
        - Pre-trained on BERT
        - Fine-tuned on SemEval 2014

        **Rule-based Model**:
        - Lexicon-based approach
        - Pattern matching
        - No training required
        """)

        # Dataset statistics
        if st.checkbox("Show Dataset Statistics"):
            st.subheader("Dataset Info")

            loader = SemEvalDataLoader("data/semeval2014_restaurants_train.csv")
            stats = loader.get_statistics()

            st.write(f"**Total Reviews**: {stats['total_reviews']}")
            st.write(f"**Total Aspects**: {stats['total_aspect_terms']}")
            st.write(f"**Avg Aspects/Review**: {stats['avg_aspects_per_review']:.2f}")

            st.write("**Sentiment Distribution**:")
            for sentiment, count in stats['sentiment_distribution'].items():
                st.write(f"- {sentiment.capitalize()}: {count}")


if __name__ == "__main__":
    main()
