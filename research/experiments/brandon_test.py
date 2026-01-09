from research.utils.data_processing import get_barra_data, filter_data
import datetime as dt

if __name__ == "__main__":
    print("hello world")
    start = dt.date(1996, 1, 1)
    end = dt.date(2024, 12, 31)
    russell = True

    columns = [
        'date',
        'barrid',
        'ticker',
        'price',
        'return',
        'market_cap', 
        'bid_ask_spread',
        'daily_volume',
        'specific_return',
        'specific_risk',
        'yield'
    ]

    data = get_barra_data(start, end, columns, russell)
    print(data)