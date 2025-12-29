"""
Main training script - Entry point for model training.
"""

import argparse
from src.training import train_model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train ABSA model')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use for training (overrides config)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Train model
    train_model(config_path=args.config)


if __name__ == "__main__":
    main()
