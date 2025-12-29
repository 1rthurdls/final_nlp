# Quick Start Guide

Get up and running with the ABSA system in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd final_version_nlp

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🎯 Try the Demo (1 minute)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser and try analyzing:

> "The food was amazing but the service was slow."

## 🏋️ Train a Model (2-3 hours)

```bash
# Quick training (3 epochs, ~30 minutes)
python train.py

# Full training (10 epochs, ~2 hours)
# Edit configs/config.yaml first, then:
python train.py --config configs/config.yaml
```

## 📊 Evaluate

```bash
python evaluate.py --model models/checkpoints/best_model.pt
```

## 🐳 Docker (30 seconds)

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

## 📚 Next Steps

1. **Explore Data**: Open `notebooks/exploration.ipynb`
2. **Run Experiments**: `python -m src.experiments`
3. **Read Full README**: See `README.md` for complete documentation
4. **Customize**: Edit `configs/config.yaml` for your needs

## ❓ Common Issues

**CUDA out of memory?**
```yaml
# In configs/config.yaml, reduce batch size:
training:
  batch_size: 8  # Instead of 16
```

**Slow training on CPU?**
```yaml
# Use smaller model:
model:
  name: "distilbert-base-uncased"
```

**Import errors?**
```bash
# Ensure you're in the right directory
pwd  # Should be .../final_version_nlp

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Open an issue on GitHub
- Email: your.email@example.com

---

**Happy analyzing! 🎯**
