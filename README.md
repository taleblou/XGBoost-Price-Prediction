# **XGBoost Model for Financial Predictions**

This repository contains an implementation of an XGBoost model, specifically designed for predicting the prices of financial instruments such as currencies, stocks, and cryptocurrencies. The XGBoost algorithm leverages gradient boosting techniques, enabling it to capture intricate patterns in price movements and handle various dataset characteristics effectively. This approach enhances the accuracy and robustness of price forecasts across various datasets.

This is the original code sample for the XGBoost model. Explore my GitHub repository for additional models and implementations that cater to different financial prediction needs.

## **Performance Metrics**

### 

### **BTC-USD (Bitcoin)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0010794601 | 0.0009626428 | 0.0010511041 | 0.0010825764 |
| Mean Absolute Error | 0.0225182790 | 0.0205550581 | 0.0226770898 | 0.0226641239 |
| R-squared | 0.9551282071 | 0.9606687535 | 0.9553621361 | 0.9556489283 |
| Median Absolute Error | 0.0145998074 | 0.0133912742 | 0.0157449225 | 0.0139960765 |
| Explained Variance Score | 0.9566631208 | 0.9621831866 | 0.9580375663 | 0.9573230196 |


### **![][image1]**

### **GC=F (Gold Futures)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0009140586 | 0.0008637735 | 0.0008719756 | 0.0009674982 |
| Mean Absolute Error | 0.0238410336 | 0.0228749418 | 0.0228566718 | 0.0246819105 |
| R-squared | 0.9537574445 | 0.9559063133 | 0.9560443828 | 0.9507698329 |
| Median Absolute Error | 0.0194905209 | 0.0160515960 | 0.0179657931 | 0.0190607390 |
| Explained Variance Score | 0.9596390659 | 0.9636102064 | 0.9624192446 | 0.9580219942 |

### **![][image2]**

### **EURUSD (Euro/US Dollar)**

| Metric | Open | High | Low | Close |
| ----- | ----- | ----- | ----- | ----- |
| Mean Squared Error | 0.0003323588 | 0.0002540997 | 0.0002617769 | 0.0003305155 |
| Mean Absolute Error | 0.0141802724 | 0.0125211860 | 0.0121102308 | 0.0141899579 |
| R-squared | 0.9292000159 | 0.9467354170 | 0.9450123323 | 0.9293981487 |
| Median Absolute Error | 0.0118845602 | 0.0108199617 | 0.0089974120 | 0.0118840140 |
| Explained Variance Score | 0.9292148194 | 0.9467440000 | 0.9452642656 | 0.9293982576 |


### 

### **![][image3]**

### **GSPC (S\&P 500 Index)**

| Metric | Open | High | Low | Close |
| :---- | :---- | :---- | :---- | :---- |
| Mean Squared Error | 0.0005145034 | 0.0004543764 | 0.0005155069 | 0.0006490753 |
| Mean Absolute Error | 0.0178120621 | 0.0168959445 | 0.0177888720 | 0.0202859200 |
| R-squared | 0.9622727325 | 0.9682476349 | 0.9618888358 | 0.9542881139 |
| Median Absolute Error | 0.0153078058 | 0.0138028655 | 0.0152938393 | 0.0167618105 |
| Explained Variance Score | 0.9691471274 | 0.9744320164 | 0.9675984899 | 0.9619588025 |

### **![][image4]**

## **Related Websites**

### [**Predict Price**](https://predict-price.com/)

Free AI-powered short-term (5/10/30 days) and long-term (6 months/1/2 years) forecasts for cryptocurrencies, stocks, ETFs, currencies, indices, and mutual funds.

### [**Magical Prediction**](https://magicalprediction.com/)

Get free trading signals generated by advanced AI models. Enhance your trading strategy with accurate, real-time market predictions powered by AI.

### [**Magical Analysis**](https://magicalanalysis.com/)

Discover free trading signals powered by expert technical analysis. Boost your forex, stock, and crypto trading strategy with real-time market insights.

## **About This Project**

This XGBoost model is an initial implementation, released for public use. The project demonstrates the potential of deep learning models for financial predictions. While this repository focuses on XGBoost, I have also utilized other models, the code for which is available on my GitHub[https://github.com/taleblou/].

## **How to Use**

1. Clone this repository.  
2. Install the required libraries: `pip install -r requirements.txt`  
3. Prepare your dataset and follow the instructions in the notebook or script.  
4. Run the model and evaluate its performance using the provided metrics.

## **License**

This project is open-source and available for public use under the MIT License. Contributions and feedback are welcome\!

[image1]: <https://raw.githubusercontent.com/taleblou/XGBoost-Price-Prediction/refs/heads/main/Plot/xgboost_BTC-USD.png>
[image2]: <https://raw.githubusercontent.com/taleblou/XGBoost-Price-Prediction/refs/heads/main/Plot/xgboost_GC%3DF.png>
[image3]: <https://raw.githubusercontent.com/taleblou/XGBoost-Price-Prediction/refs/heads/main/Plot/xgboost_EURUSD%3DX.png>
[image4]: <https://raw.githubusercontent.com/taleblou/XGBoost-Price-Prediction/refs/heads/main/Plot/xgboost_%5EGSPC.png>
