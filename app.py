from flask import Flask, request, jsonify
import pandas as pd
import cloudscraper # Swapped requests for cloudscraper
from bs4 import BeautifulSoup
import joblib
from datetime import datetime

app = Flask(__name__)
model = joblib.load('deal_model.pkl')

def check_price(url):
    try:
        # Use ScraperAPI to bypass Amazon's data center blocks
        api_key = 'b2d132d9f10293b3c7c6a138a5a6bd3e' # <-- PASTE YOUR KEY HERE
        proxy_url = f"http://api.scraperapi.com?api_key={api_key}&url={url}"
        
        # ScraperAPI handles the proxying, so we can use standard requests
        import requests
        response = requests.get(proxy_url, timeout=45)
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
            return {"title": title, "price": int(clean_price)}
        return None
    except Exception as e:
        print(f"Scraper crashed: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict_deal():
    data = request.json
    url = data.get('url')
    
    scraped_data = check_price(url)
    if not scraped_data:
        return jsonify({"error": "Failed to scrape price from URL"}), 400
        
    current_price = scraped_data['price']
    
    df = pd.read_csv('price_history.csv', parse_dates=['date'])
    
    today_row = pd.DataFrame({
        'date': [pd.to_datetime('today').normalize()],
        'title': [scraped_data['title']],
        'price': [current_price]
    })
    
    df = pd.concat([df, today_row], ignore_index=True)
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # FIX: Added min_periods=1 to prevent NaN crashes on small datasets
    df['rolling_avg_7d'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['pct_diff_from_avg'] = ((df['price'] - df['rolling_avg_7d']) / df['rolling_avg_7d']) * 100
    
    # FIX: Safety net to replace any remaining NaNs with 0
    df = df.fillna(0)
    
    today_features = df.iloc[-1][['day_of_week', 'is_weekend', 'rolling_avg_7d', 'pct_diff_from_avg']]
    X_today = today_features.values.reshape(1, -1)
    
    prediction_array = model.predict(X_today)
    prediction_result = int(prediction_array[0])
    
    return jsonify({
        "prediction": prediction_result, 
        "current_price": current_price,
        "product_title": scraped_data['title']
    })

if __name__ == '__main__':
    app.run(port=5000)