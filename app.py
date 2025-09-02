from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import numpy as np
import ta
from datetime import datetime, timedelta

app = Flask(__name__)

def get_stock_data(symbol, period='1y'):
    """株価データを取得"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        return None

def calculate_technical_indicators(df):
    """テクニカル指標を計算"""
    if df is None or len(df) < 20:
        return df
    
    # 移動平均線
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()
    
    # ボリンジャーバンド
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    return df

def calculate_return_rates(df, target_date, days_around=30):
    """指定日付を基準とした収益率を計算"""
    if df is None or len(df) == 0:
        return None
    
    try:
        target_date = pd.to_datetime(target_date)
    except:
        return None
    
    # 指定日付に最も近い営業日を見つける
    df_index = pd.to_datetime(df.index)
    closest_idx = df_index.get_indexer([target_date], method='nearest')[0]
    
    if closest_idx == -1:
        return None
    
    closest_date = df_index[closest_idx]
    base_price = df.iloc[closest_idx]['Close']
    
    # 指定日付前後のデータを取得
    start_idx = max(0, closest_idx - days_around)
    end_idx = min(len(df), closest_idx + days_around + 1)
    
    period_df = df.iloc[start_idx:end_idx].copy()
    
    # 収益率計算
    period_df['Daily_Return'] = period_df['Close'].pct_change() * 100
    period_df['Cumulative_Return'] = ((period_df['Close'] / base_price) - 1) * 100
    
    return {
        'data': period_df,
        'target_date': closest_date,
        'base_price': base_price,
        'stats': {
            '基準日価格': f"¥{base_price:.2f}",
            '期間最高収益率': f"{period_df['Cumulative_Return'].max():.2f}%",
            '期間最低収益率': f"{period_df['Cumulative_Return'].min():.2f}%",
            '平均日次収益率': f"{period_df['Daily_Return'].mean():.3f}%",
            '日次収益率標準偏差': f"{period_df['Daily_Return'].std():.3f}%"
        }
    }

def create_return_chart(return_data):
    """収益率チャートを作成"""
    if not return_data or 'data' not in return_data:
        return None
    
    df = return_data['data']
    target_date = return_data['target_date']
    
    # 累積収益率チャート
    cumulative_return = go.Scatter(
        x=df.index,
        y=df['Cumulative_Return'],
        mode='lines+markers',
        name='累積収益率',
        line=dict(color='blue')
    )
    
    # 基準日の線
    target_line = go.Scatter(
        x=[target_date, target_date],
        y=[df['Cumulative_Return'].min(), df['Cumulative_Return'].max()],
        mode='lines',
        name='基準日',
        line=dict(color='red', dash='dash')
    )
    
    # ゼロライン
    zero_line = go.Scatter(
        x=df.index,
        y=[0]*len(df),
        mode='lines',
        name='0%',
        line=dict(color='gray', dash='dot')
    )
    
    layout = go.Layout(
        title='累積収益率',
        xaxis={'title': '日付'},
        yaxis={'title': '収益率 (%)'},
        height=400
    )
    
    fig = go.Figure(data=[cumulative_return, target_line, zero_line], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_daily_return_chart(return_data):
    """日次収益率チャートを作成"""
    if not return_data or 'data' not in return_data:
        return None
    
    df = return_data['data']
    
    # 正の収益率と負の収益率で色分け
    colors = ['green' if x >= 0 else 'red' for x in df['Daily_Return'].fillna(0)]
    
    daily_return = go.Bar(
        x=df.index,
        y=df['Daily_Return'],
        name='日次収益率',
        marker=dict(color=colors)
    )
    
    layout = go.Layout(
        title='日次収益率',
        xaxis={'title': '日付'},
        yaxis={'title': '収益率 (%)'},
        height=300
    )
    
    fig = go.Figure(data=[daily_return], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_candlestick_chart(df, symbol):
    """ローソク足チャートを作成"""
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='株価'
    )
    
    # 移動平均線
    sma20 = go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA20', line=dict(color='orange'))
    sma50 = go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA50', line=dict(color='blue'))
    ema20 = go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA20', line=dict(color='green'))
    
    # ボリンジャーバンド
    bb_upper = go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(color='red', dash='dash'))
    bb_lower = go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(color='red', dash='dash'))
    bb_middle = go.Scatter(x=df.index, y=df['BB_middle'], mode='lines', name='BB Middle', line=dict(color='purple'))
    
    layout = go.Layout(
        title=f'{symbol} 株価チャート',
        xaxis={'title': '日付'},
        yaxis={'title': '価格'},
        height=500
    )
    
    fig = go.Figure(data=[candlestick, sma20, sma50, ema20, bb_upper, bb_lower, bb_middle], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_volume_chart(df):
    """出来高チャートを作成"""
    volume = go.Bar(x=df.index, y=df['Volume'], name='出来高')
    
    layout = go.Layout(
        title='出来高',
        xaxis={'title': '日付'},
        yaxis={'title': '出来高'},
        height=200
    )
    
    fig = go.Figure(data=[volume], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_rsi_chart(df):
    """RSIチャートを作成"""
    rsi = go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI')
    
    # RSIの境界線
    upper_line = go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='70', line=dict(color='red', dash='dash'))
    lower_line = go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='30', line=dict(color='red', dash='dash'))
    
    layout = go.Layout(
        title='RSI',
        xaxis={'title': '日付'},
        yaxis={'title': 'RSI', 'range': [0, 100]},
        height=200
    )
    
    fig = go.Figure(data=[rsi, upper_line, lower_line], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_macd_chart(df):
    """MACDチャートを作成"""
    macd = go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD')
    signal = go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal')
    histogram = go.Bar(x=df.index, y=df['MACD_histogram'], name='Histogram')
    
    layout = go.Layout(
        title='MACD',
        xaxis={'title': '日付'},
        yaxis={'title': 'MACD'},
        height=200
    )
    
    fig = go.Figure(data=[macd, signal, histogram], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol'].upper()
    period = request.form.get('period', '1y')
    
    # データ取得
    df = get_stock_data(symbol, period)
    if df is None or len(df) == 0:
        return jsonify({'error': '指定された銘柄のデータが見つかりません'})
    
    # テクニカル指標計算
    df = calculate_technical_indicators(df)
    
    # チャート作成
    candlestick_chart = create_candlestick_chart(df, symbol)
    volume_chart = create_volume_chart(df)
    rsi_chart = create_rsi_chart(df)
    macd_chart = create_macd_chart(df)
    
    # 統計情報
    stats = {
        '現在価格': f"¥{df['Close'][-1]:.2f}",
        '最高値': f"¥{df['High'].max():.2f}",
        '最安値': f"¥{df['Low'].min():.2f}",
        '平均出来高': f"{df['Volume'].mean():.0f}",
        '現在RSI': f"{df['RSI'][-1]:.2f}" if not pd.isna(df['RSI'][-1]) else 'N/A'
    }
    
    return jsonify({
        'candlestick_chart': candlestick_chart,
        'volume_chart': volume_chart,
        'rsi_chart': rsi_chart,
        'macd_chart': macd_chart,
        'stats': stats
    })

@app.route('/analyze_returns', methods=['POST'])
def analyze_returns():
    symbol = request.form['symbol'].upper()
    target_date = request.form['target_date']
    days_around = int(request.form.get('days_around', 30))
    
    # データ取得（長期間のデータを取得）
    df = get_stock_data(symbol, '2y')
    if df is None or len(df) == 0:
        return jsonify({'error': '指定された銘柄のデータが見つかりません'})
    
    # 収益率計算
    return_data = calculate_return_rates(df, target_date, days_around)
    if return_data is None:
        return jsonify({'error': '指定された日付のデータが見つかりません'})
    
    # チャート作成
    return_chart = create_return_chart(return_data)
    daily_return_chart = create_daily_return_chart(return_data)
    
    if return_chart is None or daily_return_chart is None:
        return jsonify({'error': 'チャートの作成に失敗しました'})
    
    return jsonify({
        'return_chart': return_chart,
        'daily_return_chart': daily_return_chart,
        'return_stats': return_data['stats'],
        'target_date': return_data['target_date'].strftime('%Y-%m-%d'),
        'base_price': return_data['base_price']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
