# filename: plot_samsung_stock.py
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 삼성전자의 티커 심볼은 "005930.KS"입니다. 미국 주식 시장 외의 주식은 종종 특정 형식의 심볼을 사용합니다.
ticker_symbol = "005930.KS"

# 오늘 날짜와 2개월 전 날짜 계산
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

# yfinance를 사용하여 주식 데이터 가져오기
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# 데이터를 pandas DataFrame으로 변환 (yfinance는 이미 DataFrame 형태로 데이터를 제공합니다)
df = pd.DataFrame(data)

# 그래프 그리기
fig = go.Figure()

# 주식 가격 데이터 추가 (종가 기준)
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', fillcolor='rgba(0,255,0,0.2)', name='Close Price'))

# y축은 구간 최솟값에서 시작하도록 설정
fig.update_yaxes(rangemode="tozero")

# 그래프 제목 및 레이아웃 설정
fig.update_layout(title='Samsung Electronics Stock Price Last 2 Months', xaxis_title='Date', yaxis_title='Price (KRW)', template='plotly_dark')

# 그래프를 이미지 파일로 저장
fig.write_image("samsung_stock_price_2m.png")

print("Samsung stock price graph has been saved as 'samsung_stock_price_2m.png'.")