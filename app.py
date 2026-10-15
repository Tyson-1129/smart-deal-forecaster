from flask import Flask, request, jsonify
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
from datetime import datetime

app = Flask(__name__)

# 1. IMPORT THE TRAINED MODEL
# Load the brain you saved from Jupyter into memory when the server starts
model = joblib.load('deal_model.pkl')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US, en;q=0.5'
}

# The exact scraper function you built earlier
def check_price(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title_element = soup.find(id='productTitle') or soup.find(class_='product-title-word-break')
    title = title_element.get_text().strip() if title_element else "Unknown Product"

    price_classes = ['a-price-whole', 'a-color-price', 'apexPriceToPay', 'priceToPay']
    price_str = None
    
    for p_class in price_classes:
        price_element = soup.find('span', class_=p_class) or soup.find('div', class_=p_class)
        if price_element:
            price_str = price_element.get_text().strip()
            break
            
    if price_str:
        clean_price = price_str.replace('₹', '').replace(',', '').replace('.00', '').replace('.', '').strip()
        try:
            return {"title": title, "price": int(clean_price)}
        except ValueError:
            return None
    return None

@app.route('/predict', methods=['POST'])
def predict_deal():
    data = request.json
    url = data.get('url')
    
    # 2. RUN THE SCRAPER ON THE URL
    scraped_data = check_price(url)
    if not scraped_data:
        return jsonify({"error": "Failed to scrape price from URL"}), 400
        
    current_price = scraped_data['price']
    
    # 3. PASS INTO PANDAS PIPELINE
    # To calculate a 7-day average for today, we must load the historical data
    df = pd.read_csv('price_history.csv', parse_dates=['date'])
    
    # Create a new row for today's scraped price
    today_row = pd.DataFrame({
        'date': [pd.to_datetime('today').normalize()],
        'title': [scraped_data['title']],
        'price': [current_price]
    })
    
    # Append today's data to the history
    df = pd.concat([df, today_row], ignore_index=True)
    
    # Run your exact feature engineering logic to generate the mathematical indicators
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['rolling_avg_7d'] = df['price'].rolling(window=7).mean()
    df['pct_diff_from_avg'] = ((df['price'] - df['rolling_avg_7d']) / df['rolling_avg_7d']) * 100
    
    # Extract ONLY the features for today (the very last row)
    today_features = df.iloc[-1][['day_of_week', 'is_weekend', 'rolling_avg_7d', 'pct_diff_from_avg']]
    
    # 4. FEED FEATURES TO SCIKIT-LEARN
    # .reshape(1, -1) is required because Scikit-learn expects a 2D array (a table of data), 
    # even when we are only passing it a single row to predict.
    X_today = today_features.values.reshape(1, -1)
    
    prediction_array = model.predict(X_today)
    prediction_result = int(prediction_array[0]) # Extracts the 0 or 1
    
    # Return the real, dynamic JSON response to your webhook
    return jsonify({
        "prediction": prediction_result, 
        "current_price": current_price,
        "product_title": scraped_data['title']
    })

if __name__ == '__main__':
    app.run(port=5000)