"""
Evaluation script for ABSA models.
"""

import argparse
import torch
import yaml
import json
from transformers import AutoTokenizer, AutoConfig

from models.advanced import BertForABSA
from src.preprocessing import create_dataloaders
from src.evaluation import evaluate_model, ErrorAnalyzer


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate ABSA model')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to test data (overrides config)'
    )

    parser.add_argument(
        '--error-analysis',
        action='store_true',
        help='Perform detailed error analysis'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load data
    test_data_path = args.data or f"{config['data']['raw_dir']}/{config['data']['test_file']}"
    print(f"Loading test data from: {test_data_path}")

    _, test_loader = create_dataloaders(
        train_path=f"{config['data']['raw_dir']}/{config['data']['train_file']}",
        test_path=test_data_path,
        tokenizer=tokenizer,
        batch_size=config['evaluation']['batch_size'],
        max_length=config['data']['max_seq_length']
    )

    # Load model
    print(f"Loading model from: {args.model}")
    bert_config = AutoConfig.from_pretrained(config['model']['name'])
    model = BertForABSA(
        bert_config,
        num_aspect_labels=config['model']['num_aspect_labels'],
        num_sentiment_labels=config['model']['num_aspect_labels']
    )

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Model loaded successfully")

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model,
        test_loader,
        device=device,
        return_predictions=args.save_predictions
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nAspect Extraction:")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall: {results.get('recall', 0):.4f}")
    print(f"  F1 Score: {results.get('f1', 0):.4f}")

    print(f"\nSentiment Classification:")
    print(f"  Precision: {results.get('sentiment_precision', 0):.4f}")
    print(f"  Recall: {results.get('sentiment_recall', 0):.4f}")
    print(f"  F1 Score: {results.get('sentiment_f1', 0):.4f}")
    print(f"  Accuracy: {results.get('sentiment_accuracy', 0):.4f}")

    # Error analysis
    if args.error_analysis:
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)

        analyzer = ErrorAnalyzer()
        # Perform error analysis on predictions
        # (This would need to be implemented based on predictions)

        summary = analyzer.get_summary()
        print(json.dumps(summary, indent=2))

        # Save error analysis
        analyzer.save_analysis('results/error_analysis.json')

    # Save results
    output_data = {
        'metrics': {k: v for k, v in results.items() if k != 'predictions'},
        'model_path': args.model,
        'config': config
    }

    if args.save_predictions and 'predictions' in results:
        output_data['predictions'] = results['predictions'][:100]  # Save first 100

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
