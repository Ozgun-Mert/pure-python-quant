import json


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _sma(values):
    """Verilen sayı listesinin basit aritmetik ortalamasını döndürür."""
    return sum(values) / len(values)


def _ema_series(values, period):
    """
    EMA serisi hesaplar.
    İlk EMA değeri, ilk `period` günün SMA'sı ile başlatılır.
    Multiplier = 2 / (period + 1)
    Hesaplanamayan günler None kalır.
    """
    n = len(values)
    ema_values = [None] * n
    if n < period:
        return ema_values

    multiplier = 2 / (period + 1)
    ema = _sma(values[:period])
    ema_values[period - 1] = ema

    for i in range(period, n):
        ema = (values[i] - ema) * multiplier + ema
        ema_values[i] = ema

    return ema_values


# ---------------------------------------------------------------------------
# Özellik ekleme fonksiyonları
# ---------------------------------------------------------------------------

def add_price_direction(data):
    """
    Bir önceki günün kapanışına göre yön bilgisi ekler.
    Artış → 1, Düşüş → -1, Aynı → 0. İlk gün → None.
    """
    for i, row in enumerate(data):
        if i == 0:
            row["Direction"] = None
            continue

        prev_close = data[i - 1]["Close"]
        curr_close = row["Close"]

        if curr_close > prev_close:
            row["Direction"] = 1
        elif curr_close < prev_close:
            row["Direction"] = -1
        else:
            row["Direction"] = 0

    return data


def add_sma_and_crossovers(data):
    """
    20, 50, 100 ve 200 günlük SMA hesaplar, ardından kesişim sinyalleri ekler.
    Cross_X_Y: SMA_X > SMA_Y ise 1, aksi halde 0.
    Yeterli veri olmayan günler None kalır.
    """
    for i, row in enumerate(data):
        # SMA_20: son 20 kapanış (bugün dahil)
        if i < 19:
            row["SMA_20"] = None
        else:
            window = [data[j]["Close"] for j in range(i - 19, i + 1)]
            row["SMA_20"] = _sma(window)

        # SMA_50: son 50 kapanış (bugün dahil)
        if i < 49:
            row["SMA_50"] = None
        else:
            window = [data[j]["Close"] for j in range(i - 49, i + 1)]
            row["SMA_50"] = _sma(window)

        # SMA_100: son 100 kapanış (bugün dahil)
        if i < 99:
            row["SMA_100"] = None
        else:
            window = [data[j]["Close"] for j in range(i - 99, i + 1)]
            row["SMA_100"] = _sma(window)

        # SMA_200: son 200 kapanış (bugün dahil)
        if i < 199:
            row["SMA_200"] = None
        else:
            window = [data[j]["Close"] for j in range(i - 199, i + 1)]
            row["SMA_200"] = _sma(window)

        # Kısa / orta / uzun vadeli crossover sinyalleri
        if row["SMA_20"] is None or row["SMA_50"] is None:
            row["Cross_20_50"] = None
        else:
            row["Cross_20_50"] = 1 if row["SMA_20"] > row["SMA_50"] else 0

        if row["SMA_50"] is None or row["SMA_100"] is None:
            row["Cross_50_100"] = None
        else:
            row["Cross_50_100"] = 1 if row["SMA_50"] > row["SMA_100"] else 0

        if row["SMA_100"] is None or row["SMA_200"] is None:
            row["Cross_100_200"] = None
        else:
            row["Cross_100_200"] = 1 if row["SMA_100"] > row["SMA_200"] else 0

    return data


def add_macd(data):
    """
    MACD indikatörünü hesaplar.
    - EMA_12 ve EMA_26: kapanış fiyatları üzerinden (SMA ile başlatılır)
    - MACD_Line: EMA_12 - EMA_26
    - MACD_Signal: MACD_Line üzerinden 9 günlük EMA (SMA ile başlatılır)
    """
    closes = [row["Close"] for row in data]
    ema_12 = _ema_series(closes, 12)
    ema_26 = _ema_series(closes, 26)

    macd_line = [None] * len(data)
    for i in range(len(data)):
        if ema_12[i] is not None and ema_26[i] is not None:
            macd_line[i] = ema_12[i] - ema_26[i]
        data[i]["EMA_12"] = ema_12[i]
        data[i]["EMA_26"] = ema_26[i]
        data[i]["MACD_Line"] = macd_line[i]

    # MACD_Signal: MACD_Line'ın 9 günlük EMA'sı
    signal_period = 9
    first_macd_idx = 25  # EMA_26'nın ilk geçerli olduğu indeks

    for i in range(len(data)):
        data[i]["MACD_Signal"] = None

    signal_start_idx = first_macd_idx + signal_period - 1
    if len(data) > signal_start_idx:
        macd_window = [macd_line[j] for j in range(first_macd_idx, signal_start_idx + 1)]
        signal = _sma(macd_window)
        data[signal_start_idx]["MACD_Signal"] = signal

        multiplier = 2 / (signal_period + 1)
        for i in range(signal_start_idx + 1, len(data)):
            signal = (macd_line[i] - signal) * multiplier + signal
            data[i]["MACD_Signal"] = signal

    return data


def add_rsi(data, period=14):
    """
    Göreceli Güç Endeksi (RSI) hesaplar.
    - İlk `period` günün kazanç/kayıpları basit ortalama ile başlatılır
    - Sonraki günlerde Wilder yumuşatılmış hareketli ortalama kullanılır
    """
    n = len(data)

    for i in range(n):
        data[i]["Gain"] = None
        data[i]["Loss"] = None
        data[i]["Avg_Gain"] = None
        data[i]["Avg_Loss"] = None
        data[i]["RSI"] = None

    # Günlük kazanç ve kayıp (ilk gün hesaplanamaz)
    for i in range(1, n):
        change = data[i]["Close"] - data[i - 1]["Close"]
        data[i]["Gain"] = change if change > 0 else 0.0
        data[i]["Loss"] = abs(change) if change < 0 else 0.0

    if n <= period:
        return data

    # İlk ortalama kazanç/kayıp: 1. günden period. güne kadar basit ortalama
    initial_gains = [data[i]["Gain"] for i in range(1, period + 1)]
    initial_losses = [data[i]["Loss"] for i in range(1, period + 1)]
    avg_gain = _sma(initial_gains)
    avg_loss = _sma(initial_losses)

    data[period]["Avg_Gain"] = avg_gain
    data[period]["Avg_Loss"] = avg_loss

    if avg_loss == 0:
        data[period]["RSI"] = 100.0
    else:
        rs = avg_gain / avg_loss
        data[period]["RSI"] = 100 - (100 / (1 + rs))

    # Sonraki günler: yumuşatılmış hareketli ortalama (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + data[i]["Gain"]) / period
        avg_loss = (avg_loss * (period - 1) + data[i]["Loss"]) / period

        data[i]["Avg_Gain"] = avg_gain
        data[i]["Avg_Loss"] = avg_loss

        if avg_loss == 0:
            data[i]["RSI"] = 100.0
        else:
            rs = avg_gain / avg_loss
            data[i]["RSI"] = 100 - (100 / (1 + rs))

    return data


def add_target_variable(data, days_ahead=3):
    """
    Model hedef değişkeni (y): days_ahead gün sonraki kapanış bugünden
    yüksekse 1, değilse 0. Son `days_ahead` güne None atanır.
    """
    n = len(data)

    for i in range(n):
        if i + days_ahead >= n:
            data[i]["Target"] = None
        else:
            future_close = data[i + days_ahead]["Close"]
            data[i]["Target"] = 1 if future_close > data[i]["Close"] else 0

    return data


def clean_matrix(data):
    """
    None içeren satırları siler ve ara hesaplama anahtarlarını temizler.
    Geriye sadece model için gerekli özellikler kalır.
    """
    intermediate_keys = ["Gain", "Loss", "Avg_Gain", "Avg_Loss", "EMA_12", "EMA_26"]

    for row in data:
        for key in intermediate_keys:
            row.pop(key, None)

    data[:] = [row for row in data if all(value is not None for value in row.values())]

    return data


def build_feature_matrix(raw_data):
    """
    Tüm özellik mühendisliği adımlarını sırayla çalıştırır,
    temizlenmiş matrisi final_matrix.json olarak kaydeder.
    """
    add_price_direction(raw_data)
    add_sma_and_crossovers(raw_data)
    add_macd(raw_data)
    add_rsi(raw_data)
    add_target_variable(raw_data)
    clean_matrix(raw_data)

    with open("final_matrix.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    return raw_data


if __name__ == "__main__":
    with open("aapl_data.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    matrix = build_feature_matrix(raw_data)
    print(f"Feature matrix ready: {len(matrix)} rows saved to final_matrix.json")

    if matrix:
        print("Sample row:", matrix[0])
