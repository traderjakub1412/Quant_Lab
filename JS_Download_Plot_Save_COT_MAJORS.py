import pandas as pd
import yfinance as yf
import cot_reports as cot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Usage
start_year = 2020
end_year = 20240
start_year = int(input("Enter the start year: "))
end_year = int(input("Enter the end year: "))


# COT Index Calculation

def calculate_cot_index(series):
    low = series.min()
    high = series.max()
    current_week = series.iloc[-1]

    cot_index = (current_week - low) / (high - low)
    cot_index = round(cot_index * 100, 1)

    return cot_index

# Download COT data only once
def download_cot_data(start_year, end_year):
    dfs = []
    for i in range(start_year, end_year + 1):
        single_year = pd.DataFrame(cot.cot_year(i, cot_report_type='legacy_fut'))
        dfs.append(single_year)

    df_cot = pd.concat(dfs)
    return df_cot

# Function to analyze COT data and price data for a specific contract
def analyze_contract(df_cot, contract_code, instrument_code, start_year, end_year, save_csv=False):
    # Filter COT data for the specific contract
    asset_cot = df_cot[df_cot['CFTC Contract Market Code (Quotes)'] == contract_code]
    asset_cot = asset_cot[['As of Date in Form YYYY-MM-DD', 'Market and Exchange Names','CFTC Contract Market Code',
                           'Open Interest (All)', 'Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
                           'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)',
                           'Nonreportable Positions-Long (All)', 'Nonreportable Positions-Short (All)']].copy()

    
    # Net Positions
    asset_cot['Net_Position_NonComm'] = asset_cot['Noncommercial Positions-Long (All)'] - asset_cot['Noncommercial Positions-Short (All)']
    asset_cot['Net_Position_Comm'] = asset_cot['Commercial Positions-Long (All)'] - asset_cot['Commercial Positions-Short (All)']
    asset_cot['Net_Position_NonRept'] = asset_cot['Nonreportable Positions-Long (All)'] - asset_cot['Nonreportable Positions-Short (All)']

    # Datetime and index
    asset_cot['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(asset_cot['As of Date in Form YYYY-MM-DD'])
    asset_cot = asset_cot.set_index('As of Date in Form YYYY-MM-DD').sort_index()

    # COT Index for noncom, com, retail and oi

    asset_cot['COT_Index_NonComm'] = asset_cot['Net_Position_NonComm'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_Comm'] = asset_cot['Net_Position_Comm'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_NonRept'] = asset_cot['Net_Position_NonRept'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_OI'] = asset_cot['Open Interest (All)'].rolling(26).apply(calculate_cot_index)
    asset_cot['OI_Percentage_Comm'] = (asset_cot['Net_Position_Comm'] / asset_cot['Open Interest (All)'])
    asset_cot['OI_Percentage_Comm'] = asset_cot['OI_Percentage_Comm'].rolling(26).apply(calculate_cot_index)
    asset_cot['OI_Percentage_Short'] = (asset_cot['Commercial Positions-Short (All)'] /asset_cot['Open Interest (All)'])
    asset_cot['OI_Percentage_Short'] = asset_cot['OI_Percentage_Short'].rolling(26).apply(calculate_cot_index)

    # Price data
    asset_price = yf.download(instrument_code, start=f"{start_year}-01-01", end=f"{end_year}-12-31", progress=False)

    # Merged DataFrame
    merged_df = pd.merge_asof(asset_price, asset_cot, left_index=True, right_index=True, direction='nearest')
    
    columns_to_shift = ['Open Interest (All)',
    'Noncommercial Positions-Long (All)',
    'Noncommercial Positions-Short (All)',
    'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)',
    'Nonreportable Positions-Long (All)',
    'Nonreportable Positions-Short (All)', 'Net_Position_NonComm',
    'Net_Position_Comm', 'Net_Position_NonRept', 'COT_Index_NonComm',
    'COT_Index_Comm', 'COT_Index_NonRept', 'COT_Index_OI']

    merged_df[columns_to_shift] = merged_df[columns_to_shift].shift(5)
    
    
    # Decompose time series into trend, seasonal, and residual components
    result = seasonal_decompose(merged_df['Close'].dropna(), model='additive', period=252)  # Assuming daily data with a yearly cycle

    # Extract seasonality and fill NaN values
    merged_df = merged_df.copy()
    merged_df['Seasonality'] = result.seasonal

    merged_df = merged_df.copy()
    merged_df['Seasonality'].fillna(0, inplace=True)


    if save_csv:
        csv_filename = f"{instrument_code}_{contract_code}_{start_year}_{end_year}_merged.csv"
        merged_df.to_csv(csv_filename, index=True)
        print(f"CSV file saved as {csv_filename}")

    return merged_df

# Function to analyze all contracts using the downloaded COT data
def analyze_all_contracts(df_cot, contracts_info, start_year, end_year, save_csv=True):
    merged_dfs = []
    
    for contract_info in contracts_info:
        contract_code = contract_info['contract_code']
        instrument_code = contract_info['instrument_code']
        merged_df = analyze_contract(df_cot, contract_code, instrument_code, start_year, end_year, save_csv=False)
        merged_dfs.append(merged_df)

    # Concatenate all merged DataFrames
    final_merged_df = pd.concat(merged_dfs)

    if save_csv:
        csv_filename = f"all_contracts_{start_year}_{end_year}_merged.csv"
        final_merged_df.to_csv(csv_filename, index=True)
        print(f"CSV file saved as {csv_filename}")

    return final_merged_df


# Download COT data only once
df_cot_data = download_cot_data(start_year, end_year)

contracts_info = [
    {'contract_code': '099741', 'instrument_code': '6E=F'},  # Euro FX
    {'contract_code': '096742', 'instrument_code': '6B=F'},  # British Pound
    {'contract_code': '090741', 'instrument_code': '6C=F'},  # Canadian Dollar
    {'contract_code': '092741', 'instrument_code': '6S=F'},  # Swiss Franc
    {'contract_code': '097741', 'instrument_code': '6J=F'},  # Japanese Yen
    {'contract_code': '112741', 'instrument_code': '6N=F'},  # NZ Dollar
    {'contract_code': '232741', 'instrument_code': '6A=F'}  # Australian Dollar
]

# Analyze all contracts using the downloaded COT data
analyze_all_contracts(df_cot_data, contracts_info, start_year, end_year, save_csv=True)
print('Data Adjustements DONE :) ')

df = pd.read_csv(f"all_contracts_{start_year}_{end_year}_merged.csv",index_col=0,parse_dates=True)

dfpl = df[:]

# Function to find the most recent Friday
def last_friday():
    today = datetime.now()
    days_to_friday = (4 - today.weekday() + 7) % 7
    last_friday_date = today - timedelta(days=days_to_friday)
    return last_friday_date

# Create a folder named 'Reports' if it doesn't exist
folder_path = 'Reports'
os.makedirs(folder_path, exist_ok=True)

unique_assets = [99741, 96742, 90741, 92741, 97741, 112741, 232741]

for asset in unique_assets:
    asset_df = dfpl[dfpl['CFTC Contract Market Code'] == asset]
    
    # Create subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[5, 2, 1,3])

   # Add traces to the subplots
    fig.add_trace(go.Candlestick(x=asset_df.index,
                                open=asset_df['Open'],
                                high=asset_df['High'],
                                low=asset_df['Low'],
                                close=asset_df['Close']), row=1, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_Comm'], mode='lines', name=f'Commercial: {asset_df["Net_Position_Comm"].iloc[-1]}', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_NonComm'], mode='lines', name=f'Non-Commercial: {asset_df["Net_Position_NonComm"].iloc[-1]}', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_NonRept'], mode='lines', name=f'Non-Reportable: {asset_df["Net_Position_NonRept"].iloc[-1]}', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Open Interest (All)'], mode='lines', name=f'Open Interest: {asset_df['Open Interest (All)'].iloc[-1]}', line=dict(color='black')), row=2, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_Comm'], mode='lines', name=f'Commercial Index: {asset_df["COT_Index_Comm"].iloc[-1]}%', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_NonComm'], mode='lines', name=f'Non-Commercial Index: {asset_df["COT_Index_NonComm"].iloc[-1]}%', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_NonRept'], mode='lines', name=f'Non-Reportable Index: {asset_df["COT_Index_NonRept"].iloc[-1]}%', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_OI'], mode='lines', name=f'Open Interest Index: {asset_df["COT_Index_OI"].iloc[-1]}%', line=dict(color='black')), row=3, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Seasonality'], mode='lines', name='Seasonality', line=dict(color='green')), row=4, col=1)

    # Add yearly vertical lines
    for year in df.index.year.unique():
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=f"{year}-01-01",
                x1=f"{year}-01-01",
                y0=asset_df['Low'].min(),
                y1=asset_df['High'].max(),
                line=dict(color="grey", width=1),
            )
        )
        
    # Update layout with specified height and width
    fig.update_layout(
        title_text=asset_df['Market and Exchange Names'].iloc[-1],
        showlegend=True,
        height=1000,
        width=1800,
        hovermode='x unified'
    )
    fig.update_traces(xaxis='x')

    fig.update_xaxes(rangeslider_visible=False, range=[asset_df.index[0], asset_df.index[-1] + pd.DateOffset(days=50)])

    
    # Get the last Friday's date
    last_friday_date = last_friday()

    
    # Format the date as 'YYYY-MM-DD'
    formatted_date = last_friday_date.strftime('%Y-%m-%d')

    # Save the figure in the 'Reports' folder with date in the file name
    fig_file = os.path.join(folder_path, f"{formatted_date}_{asset_df['Market and Exchange Names'].iloc[-1]}_figure.html")
    pio.write_html(fig, file=fig_file, auto_open=False)
    print(f"{fig_file} saved :) ")

print('Job DONE :) ')