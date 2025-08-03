# 🚗 Car Price Predictor

A Streamlit web application that predicts fair market prices for vehicles based on their characteristics. Currently supports 2026 Hyundai Palisade with plans to expand to other models.

## 🎯 Features

- **Price Prediction**: Uses a trained machine learning model to predict fair market prices
- **Real-time Analysis**: Compare asking prices against predicted values
- **Visual Feedback**: Interactive charts and clear price evaluations
- **User-friendly Interface**: Clean, intuitive interface for easy price checking

## 🚀 Live Demo

[Visit the live app on Streamlit Cloud](https://car-pricer.streamlit.app)

## 📋 How It Works

1. **Select Vehicle Details**: Choose trim level, drivetrain, city MPG, and color options
2. **Enter Asking Price**: Input the price you want to evaluate
3. **Get Prediction**: Click "Predict Price" to see the analysis
4. **Evaluate Deal**: See if the price is fair, overpriced, or a good deal

## 🔧 Current Model: 2026 Hyundai Palisade

The prediction model uses these vehicle characteristics:

- **Trim Levels**: Calligraphy, Limited, SEL, SEL Convenience
- **Drivetrain**: AWD, FWD
- **City MPG**: 15-25 range
- **Exterior Colors**: Abyss Black, Classy Blue, Creamy White, Ecotronic Gray
- **Interior Colors**: Black, Brown, Gray, Gray/Navy, Navy/Brown

*More vehicle models coming soon!*

## 📊 Price Evaluation

- ✅ **Fair Price**: Within 5% of predicted value
- ⚠️ **Overpriced**: More than 5% above predicted value
- 💰 **Good Deal**: More than 5% below predicted value

## 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/cpkelley5/car-price-predictor.git
cd car-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
car-price-predictor/
├── app.py                      # Main Streamlit application
├── palisade_price_model.pkl    # Trained ML model (Palisade)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── run_app.sh                  # Local run script
```

## 🤖 Model Information

### Current Model: 2026 Hyundai Palisade

The prediction model was trained on 2026 Hyundai Palisade data using scikit-learn. It uses one-hot encoding for categorical features and includes:

- 4 trim level features
- 2 drivetrain features  
- 1 numerical feature (City MPG)
- 4 exterior color features
- 5 interior color features

**Total**: 16 features for prediction

## 🔄 Version Compatibility

The app handles model loading gracefully:
- Tries `joblib.load()` first (recommended)
- Falls back to `pickle.load()` if needed
- Provides demo mode if model loading fails

## 📝 License

This project is for educational and personal use. The ML model is trained on publicly available automotive data.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Built with ❤️ using [Streamlit](https://streamlit.io/)