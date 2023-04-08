import pandas as pd
import numpy as np


def wrangle(file_path):
    df_original = pd.read_csv(file_path)
    # Create short simple moving average over the short window
    df_original['short_mavg'] = df_original['close'].rolling(
        window=10, min_periods=1, center=False).mean()
    # Create long simple moving average over the long window
    df_original['long_mavg'] = df_original['close'].rolling(
        window=60, min_periods=1, center=False).mean()

    # Create signals
    df_original['signal'] = np.where(
        df_original['short_mavg'] > df_original['long_mavg'], 1.0, 0.0)
    return df


df_original = wrangle()

# calculation of exponential moving average


def EMA(df, n):
    EMA = pd.Series(df['close'].ewm(
        span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA


df_original['EMA10'] = EMA(df_original, 10)
df_original['EMA30'] = EMA(df_original, 30)
df_original['EMA200'] = EMA(df_original, 200)
df_original.head()

# calculation of rate of change


def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC


df_original['ROC10'] = ROC(df_original['close'], 10)
df_original['ROC30'] = ROC(df_original['close'], 30)

# Calculation of price momentum


def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM


df_original['MOM10'] = MOM(df_original['close'], 10)
df_original['MOM30'] = MOM(df_original['close'], 30)

# calculation of relative strength index


def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[period-1]] = np.mean(u[:period])
    u = u.drop(u.index[:(period-1)])
    # first value is sum of avg losses
    d[d.index[period-1]] = np.mean(d[:period])
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / \
        d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


df_original['RSI10'] = RSI(df_original['close'], 10)
df_original['RSI30'] = RSI(df_original['close'], 30)
df_original['RSI200'] = RSI(df_original['close'], 200)

# calculation of stochastic osillator.


def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) /
            (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK


def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) /
            (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD


df_original['%K10'] = STOK(
    df_original['close'], df_original['low'], df_original['high'], 10)
df_original['%D10'] = STOD(
    df_original['close'], df_original['low'], df_original['high'], 10)
df_original['%K30'] = STOK(
    df_original['close'], df_original['low'], df_original['high'], 30)
df_original['%D30'] = STOD(
    df_original['close'], df_original['low'], df_original['high'], 30)
df_original['%K200'] = STOK(
    df_original['close'], df_original['low'], df_original['high'], 200)
df_original['%D200'] = STOD(
    df_original['close'], df_original['low'], df_original['high'], 200)


def MA(df, n):
    MA = pd.Series(df['close'].rolling(
        n, min_periods=n).mean(), name='MA_' + str(n))
    return MA


df_original['MA21'] = MA(df_original, 10)
df_original['MA63'] = MA(df_original, 30)
df_original['MA252'] = MA(df_original, 200)
df_original.tail()


drop = ['high', 'low', 'open', 'close_time', 'quote_asset_volume', 'taker_buy_base_asset_volume',
        'number_of_trades', 'taker_buy_quote_asset_volume', 'ignore', 'timestamp', 'short_mavg', 'long_mavg']


df_original.drop(columns=drop, inplace=True)
print(f'Data Shape--> {df_original.shape}')


df_original = df_original.dropna(axis=0)
