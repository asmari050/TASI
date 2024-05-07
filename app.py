import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc 
import re
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from zigzag import peak_valley_pivots
import dash_html_components as html
# إنشاء التطبيق الرئيسي مع استخدام رابط external_stylesheets لـ Bootstrap و تجاهل الاستثناءات للمكونات غير الموجودة في التخطيط الأولي
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

server = app.server
df_sr = pd.read_csv('Name/SR.CSV')


# تعريف تخطيط التطبيق الرئيسي
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Stock Data KSA "),
                dbc.Button("مؤشر السيولة ", color="success", className="mt-3 btn-block", href="/app1", style={'margin-bottom': '10px'}),
                dbc.Button(" نظام SK للأسهم", color="warning", className="mt-3 btn-block", href="/app2", style={'margin-bottom': '10px'}),
                dbc.Button("التطبيق الثالث", color="primary", className="mt-3 btn-block", href="/app3", style={'margin-bottom': '10px'}),
                dbc.Button("التطبيق الرابع", color="primary", className="mt-3 btn-block", href="/app4", style={'margin-bottom': '10px'}),
                dbc.Button(" قائمة الأسهم", color="primary", className="mt-3 btn-block", href="/app5", style={'margin-bottom': '10px'}),
            
            ], width=60)
        ])
    ]),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

options = [{'label': f"{row['name_stock']} ({row['stock_symbol']})", 'value': row['stock_symbol']} for index, row in df_sr.iterrows()]

# تعريف تخطيطات التطبيقات الداخلية
app1_layout = html.Div([
        html.Hr(),
        html.Div([
           
            html.Ul([
                   html.Li([
                        html.Label("أدخل الرمز :"),
                        dcc.Input(
                                id='symbol_input',
                                type='text',
                                value='TASI',
                                debounce=True,
                                maxLength=4,)  # Limits the user to input no more than 4 characters
                    ]),

                    html.Li([
                          html.Label(" أو اختر الشركة :"),
                            dcc.Dropdown(
                            id='symbol_input1',
                            options=options,
                            value='TASI',  # Default value
                            searchable=True,
                            style={'color': 'black', 'fontSize': '14px', 'width': '300px'},
                            ),
                                                
                            ]),

                    html.Li([
                        html.Label("الفترة الزمنية :"),
                        dcc.Dropdown(
                            id='bar_size_combo',
                            options=[
                                {'label': 'ساعة', 'value': '1h'},
                                {'label': 'يومي', 'value': '1d'},
                                {'label': 'أسبوعي', 'value': '1wk'},
                                {'label': 'شهري', 'value': '1mo'},
                                {'label': 'ربع سنوي', 'value': '3mo'}
                            ],
                            value='1d'
                        ),
                        
                    ]),
            ]),
            
            #===================معلومات الشركة===============================================================
            html.Ul([
                   html.Table(
                                children=[
                                    html.Caption('معلومات الشركة', style={'background-color': '#c0bebe','border': '1px solid #9b9898','caption-side': 'top', 'font-size': '26px', 'color': 'black', 'text-align': 'center','padding':'10px','font-family':'Calibri','fontSize': '22px'}),
                                    
                                    html.Tbody(id='data_table')
                                ],
                            )
                ]),
            
            html.Ul([
                    html.Div([
                            dcc.Graph(id='candlestick_graph', style={'width': '1100px', 'height': '600px', 'margin': 'auto'})
                    ])
                     
                ]),
        ],className='input_S1')
    ])
#====================الجدول والادخال  =================
@app.callback(
    Output('data_table', 'children'),
    [Input('symbol_input', 'value'), Input('bar_size_combo', 'value')]
)
def update_data_table(symbol, bar_size):
    if not symbol or (len(symbol) != 4 and symbol.upper() != "TASI"):
        raise dash.exceptions.PreventUpdate

    if symbol.upper() == "TASI":
        symbol = "^TASI.SR"
        stock_name = "المؤشر العام للسوق السعودي"
    else:
        symbol = symbol + ".SR" if not symbol.endswith(".SR") else symbol
        matched_row = df_sr[df_sr['stock_symbol'] == symbol]
        stock_name = matched_row['name_stock'].iloc[0] if not matched_row.empty else symbol

    if bar_size == '1h':
        period = '1mo'
    elif bar_size == '1d':
        period = '2mo'
    elif bar_size == '1wk':
        period = '2y'
    elif bar_size == '1mo':
        period = '5y'
    elif bar_size == '3mo':
        period = '10y'

    data = yf.download(symbol, period=period, interval=bar_size).copy()


    if data.empty:
        raise dash.exceptions.PreventUpdate

    last_day_data = data.iloc[-1:].copy()
    data['Index'] = (data['Low'] >= data['Low'].shift(-1)) & \
                    (data['Low'] >= data['Low'].shift(-2)) & \
                    (data['Low'] >= data['Low'].shift(2)) & \
                    (data['Low'] >= data['Low'].shift(3))

    data['i'] = np.where(data['Index'].shift(-1), data['High'], np.nan).copy()

    if not data['i'].empty:
        data['i'].iloc[-1] = np.nan
    # إضافة عمود last_i_value
    last_i_value = data['i'].dropna().iloc[-1] if not data['i'].dropna().empty else np.nan
    last_i_date = data.index[data['i'] == last_i_value][-1] if not pd.isnull(last_i_value) else None
    # Get the last day data only
    last_day_data = data.iloc[-1]

    # Calculate the change percentage for the last day
    change = ((last_day_data['Close'] - last_day_data['Open']) / last_day_data['Open']) * 100 if last_day_data['Open'] != 0 else 0

    # Create a separate row for each piece of data for the last day
    table_rows = [
        html.Tr([html.Td("الاسم ", className='T'), html.Td(stock_name, className='TT')]),
        html.Tr([html.Td("الرمز", className='T'), html.Td(symbol.replace(".SR", "").replace("^", ""), className='TT')]),
        html.Tr([html.Td("التاريخ", className='T'), html.Td(last_day_data.name.strftime('%Y-%m-%d'), className='TT')]),
        html.Tr([html.Td("الإغلاق", className='T'), html.Td(f"{last_day_data['Close']:.2f}", className='TT')]),
        html.Tr([html.Td("الافتتاح", className='T'), html.Td(f"{last_day_data['Open']:.2f}", className='TT')]),
        html.Tr([html.Td("الأعلى", className='T'), html.Td(f"{last_day_data['High']:.2f}", className='TT')]),
        html.Tr([html.Td("الأدنى", className='T'), html.Td(f"{last_day_data['Low']:.2f}", className='TT')]),
        html.Tr([html.Td("التغيير", className='T'), html.Td(f"{change:.2f}%", className='TT')]),
        html.Tr([html.Td("السيولة", className='T'), html.Td(f"{last_i_value:.2f}", className='TT')])
    ]

    return table_rows

#-------------------------------------------------------
#====================الجدول القائمة  =================



#====================الرسم البياني =================
@app.callback(
    Output('candlestick_graph', 'figure'),
    [Input('symbol_input', 'value'), Input('bar_size_combo', 'value')]
)
def update_outputs(symbol, bar_size):
    if not symbol or (len(symbol) != 4 and symbol.upper() != "TASI"):
        raise dash.exceptions.PreventUpdate

    if symbol.upper() == "TASI":
        symbol = "^TASI.SR"
        stock_name = "المؤشر العام للسوق السعودي"
    else:
        symbol = symbol + ".SR" if not symbol.endswith(".SR") else symbol
        matched_row = df_sr[df_sr['stock_symbol'] == symbol]
        stock_name = matched_row['name_stock'].iloc[0] if not matched_row.empty else symbol

    if bar_size == '1h':
        period = '1mo'
    elif bar_size == '1d':
        period = '3mo'
    elif bar_size == '1wk':
        period = '2y'
    elif bar_size == '1mo':
        period = '5y'
    elif bar_size == '3mo':
        period = '10y'

    data = yf.download(symbol, period=period, interval=bar_size).copy()

    if data.empty:
        raise dash.exceptions.PreventUpdate

    data['Index'] = (data['Low'] >= data['Low'].shift(-1)) & \
                    (data['Low'] >= data['Low'].shift(-2)) & \
                    (data['Low'] >= data['Low'].shift(2)) & \
                    (data['Low'] >= data['Low'].shift(3))

    data['i'] = np.where(data['Index'].shift(-1), data['High'], np.nan).copy()

    if not data['i'].empty:
        data['i'].iloc[-1] = np.nan

    data['MA20'] = data['Close'].rolling(window=20).mean()

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#008001',
        decreasing_line_color='red',
        increasing_fillcolor='#008001',
        decreasing_fillcolor='red',
        hoverinfo='none',
        text=[f"Open: {o}<br>High: {h}<br>Low: {l}<br>Close: {c}" for o, h, l, c in zip(data['Open'], data['High'], data['Low'], data['Close'])]
    )])
    

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['i'],
        
        mode='markers',
        marker=dict(color='green', size=6),
        name='i'
    )),



    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        mode='lines',
        line=dict(color='#0b5ed7', width=3),
        name='MA20'
    ))

    if not data['i'].dropna().empty:
        last_i_value = data['i'].dropna().iloc[-1]
        last_i_date = data.index[data['i'] == last_i_value][-1]

        fig.add_shape(
            type="line",
            x0=last_i_date, y0=last_i_value,
            x1=data.index[-1], y1=last_i_value.max(),
            line=dict(color='#0e0d0d', width=3),
            xref="x", yref="y"
        )

        fig.add_annotation(
            x=last_i_date,
            y=last_i_value.max(),
            text=f"{last_i_value.max():.2f} خط السيولة",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#0e0d0d",
            arrowsize=1,
            arrowwidth=2,
            font=dict(color="#0e0d0d", size=19, family="Dubai"),  # تحديد نوع الخط
            xref="x",
            yref="y",
            yshift=10
        )

    bar_size_text = ""
    if bar_size == '1h':
        bar_size_text = "ساعة"
    elif bar_size == '1d':
        bar_size_text = "يومي"
    elif bar_size == '1wk':
        bar_size_text = "أسبوعي"
    elif bar_size == '1mo':
        bar_size_text = "شهري"
    elif bar_size == '3mo':
        bar_size_text = "ربع سنوي"

    fig.update_layout(
        title=f'<b>الرسم البياني {stock_name} - {symbol.replace(".SR", "")} - {bar_size_text}</b>',
        title_x=0.5,
        title_font=dict(size=22, color='#0e0d0d'),  # لون العنوان باللون الأبيض
        yaxis_title='السعر',
        yaxis_title_font=dict(size=22, color='#0e0d0d'),  # لون عنوان محور y باللون الأبيض
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', showgrid=False, showticklabels=False),
        yaxis=dict(
            tickformat=".2f",
            tickfont=dict(color='#0e0d0d', size=16),  # لون وحجم الأرقام باللون الأبيض والحجم 22
            showgrid=False,  # إخفاء الخطوط العرضية
            showline=False,  # إخفاء الخطوط الرأسية
            ticklen=100,  # زيادة المسافة بين الشموع والأرقام
        ),
        plot_bgcolor='#f0f0f1',  # لون الخلفية للرسم البياني
        paper_bgcolor='#f0f0f1',  # لون الخلفية للمنطقة الرسم بالكامل
        yaxis_side='right',
        yaxis2=dict(
            title='السعر',
            title_font=dict(size=22, color='#0e0d0d'),  # لون عنوان محور y الثاني باللون الأبيض
            overlaying='y',
            side='right',
            showline=False,  # إزالة الخط العرضي
            tickfont=dict(color='#0e0d0d', size=16),  # لون وحجم الأرقام باللون الأبيض والحجم 22
            ticklen=300,  # زيادة المسافة بين الشموع والأرقام
        ),
    )

    return fig
#-------------------------------------------------------


        


app2_layout = html.Div([
        html.Hr(),
        html.Div([
           
            html.Ul([
                   html.Li([
                        html.Label("أدخل الرمز :"),
                        dcc.Input(
                                id='symbol_input',
                                type='text',
                                value='TASI',
                                debounce=True,
                                maxLength=4,)  # Limits the user to input no more than 4 characters
                    ]),

                    html.Li([
                        html.Label("الفترة الزمنية :"),
                        dcc.Dropdown(
                            id='bar_size_combo',
                            options=[
                               
                                {'label': 'يومي', 'value': '1d'},
                                {'label': 'أسبوعي', 'value': '1wk'},
                                {'label': 'شهري', 'value': '1mo'},
                                
                            ],
                            value='1d'
                        ),
                        
                    ]),
            ]),
            
            #===================معلومات الشركة===============================================================
            html.Div([
                    html.Ul([
                   html.Table(
                                children=[
                                    html.Caption('معلومات الاختراق', style={'background-color': '#c0bebe','border': '1px solid #9b9898','caption-side': 'top', 'font-size': '26px', 'color': 'black', 'text-align': 'center','padding':'10px','font-family':'Calibri','fontSize': '22px'}),
                                    
                                    html.Tbody(id='data_table2')
                                ],
                            )
                ]),
            
            html.Ul([
                    html.Div([
                           dcc.Graph(id='candlestick_zigzag_graph', style={'width': '1100px', 'height': '600px', 'margin': 'auto'})
                    ])
                     
                ]),

            ])
            
        ],className='input_S1')
    ])

# تحديث callback لرسم الشموع مع نقاط الزقزاق
import dash_html_components as html

# تحديث callback لرسم الشموع مع نقاط الزقزاق وإنشاء الجدول
@app.callback(
    [Output('candlestick_zigzag_graph', 'figure'),
     Output('data_table2', 'children')],
    [Input('symbol_input', 'value'),
     Input('bar_size_combo', 'value')]
)
def update_zigzag_outputs(symbol, bar_size):
    # قم بالتعديلات اللازمة في هذا الجزء

    # تحقق من قيمة `data_table` قبل الإشارة إليها
    data_table = None

    # اختبر الرمز والحجم الزمني للشريط
    if not symbol or (len(symbol) != 4 and symbol.upper() != "TASI"):
        raise dash.exceptions.PreventUpdate

    if symbol.upper() == "TASI":
        symbol = "^TASI.SR"
        stock_name = "المؤشر العام للسوق السعودي"
    else:
        symbol = symbol + ".SR" if not symbol.endswith(".SR") else symbol
        matched_row = df_sr[df_sr['stock_symbol'] == symbol]
        stock_name = matched_row['name_stock'].iloc[0] if not matched_row.empty else symbol

    
    if bar_size == '1d':
        period = '3mo'
    elif bar_size == '1wk':
        period = '2y'
    elif bar_size == '1mo':
        period = '5y'
    

    data = yf.download(symbol, period=period, interval=bar_size).copy()

    if data.empty:
        raise dash.exceptions.PreventUpdate

    # Calculate Zigzag points
    pivotsh = peak_valley_pivots(data['High'].to_numpy(), 0.03, -0.03)
    pivotsl = peak_valley_pivots(data['Low'].to_numpy(), 0.03, -0.03)
    latest_peaks = data.loc[pivotsh == 1, 'High'].tail(6)
    latest_valleys = data.loc[pivotsl == -1, 'Low'].tail(6)

    # Create candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#008001',
        decreasing_line_color='red',
        increasing_fillcolor='#008001',
        decreasing_fillcolor='red',
        hoverinfo='none',
        name='Candlestick'
    ))

# Assuming latest_peaks and latest_valleys are defined elsewhere

    # Add Zigzag points to the chart
    fig.add_trace(go.Scatter(
        x=latest_peaks.index,
        y=latest_peaks,
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Latest Peaks'
    ))
    fig.add_trace(go.Scatter(
        x=latest_valleys.index,
        y=latest_valleys,
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Latest Valleys'
    ))

    # Connect the points with a line
    # Combine the latest peaks and valleys, sort them, and remove duplicates
    connected_points = pd.concat([latest_peaks, latest_valleys]).drop_duplicates().sort_index()

    # Add the line connecting the points
    fig.add_trace(go.Scatter(
        x=connected_points.index,
        y=connected_points,
        mode='lines',
        line=dict(color='black', width=2),
        name='Connected Line'
    ))

    # Check if the conditions are met before displaying the value
    if len(connected_points) >= 4:
        if (connected_points.iloc[-1] > connected_points.iloc[-2] 
            and connected_points.iloc[-3] > connected_points.iloc[-4] 
            and connected_points.iloc[-2] > connected_points.iloc[-4]):
            last_third_value = format(connected_points.iloc[-3], '.2f')
            
            # Define the third point and last zigzag point for the straight line
            third_point = connected_points.index[-3]
            last_zigzag_point = connected_points.loc[third_point]
                
            # Define start and end points for the straight line
            start_point = (third_point, last_zigzag_point)
            end_point = (data.index[-1], last_zigzag_point)  # End point at the same y-value as the third point

            # Add the straight line
            fig.add_trace(go.Scatter(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                name='Straight Line'
            ))
        else:
            last_third_value = ""  # تعيين قيمة فارغة إذا لم يتحقق الشرط
    else:
        last_third_value = ""  # تعيين قيمة فارغة إذا لم تتوفر البيانات
    # Check if the conditions are met before displaying the value
    if len(connected_points) >= 5:
        if (connected_points.iloc[-2] > connected_points.iloc[-3] 
            and connected_points.iloc[-4] > connected_points.iloc[-3] 
            and connected_points.iloc[-4] > connected_points.iloc[-5] 
            and connected_points.iloc[-3] > connected_points.iloc[-5]):
            last_third_value = format(connected_points.iloc[-4], '.2f')
            
            # Define the third point and last zigzag point for the straight line
            third_point = connected_points.index[-4]
            last_zigzag_point = connected_points.loc[third_point]
                
            # Define start and end points for the straight line
            start_point = (third_point, last_zigzag_point)
            end_point = (data.index[-1], last_zigzag_point)  # End point at the same y-value as the third point

            # Add the straight line
            fig.add_trace(go.Scatter(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                name='Straight Line'
            ))
        else:
            last_third_value = ""  # تعيين قيمة فارغة إذا لم يتحقق الشرط
    else:
        last_third_value = ""  # تعيين قيمة فارغة إذا لم تتوفر البيانات
        # Build the data table with the latest six zigzag points
    data_table = html.Table(
        [
            *html.Tr([html.Th("التاريخ"), html.Th("سعر الدخول")]),
            *([html.Tr([html.Td(connected_points.index[-3].strftime('%Y-%m-%d')), html.Td(last_third_value)])] )
        ],
        style={'margin': 'auto', 'width': '300px', 'border': '1px solid black'}
    )

  


    # Customize the layout of the chart
    fig.update_layout(
        title=f'<b>{stock_name} - {symbol.replace(".SR", "")} - {bar_size}</b>',
        title_x=0.5,
        title_font=dict(size=22, color='#0e0d0d'),
        yaxis_title='Price',
        yaxis_title_font=dict(size=22, color='#0e0d0d'),
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', showgrid=False, showticklabels=False),
        yaxis=dict(
            tickformat=".2f",
            tickfont=dict(color='#0e0d0d', size=16),
            showgrid=False,
            showline=False,
            ticklen=100,
        ),
        plot_bgcolor='#f0f0f1',
        paper_bgcolor='#f0f0f1',
        yaxis_side='right',
        yaxis2=dict(
            title='Price',
            title_font=dict(size=22, color='#0e0d0d'),
            overlaying='y',
            side='right',
            showline=False,
            tickfont=dict(color='#0e0d0d', size=16),
            ticklen=300,
        ),
    )

    return fig, data_table





app3_layout = html.Div([
    html.H2("التطبيق الثالث"),
    html.P("محتوى التطبيق الثالث يمكن وضعه هنا..."),
])

app4_layout = html.Div([
    html.H2("التطبيق الرابع"),
    html.P("محتوى التطبيق الرابع يمكن وضعه هنا..."),
])

app5_layout = html.Div([
    html.H2("التطبيق الخامس"),
    html.P("محتوى التطبيق الخامس يمكن وضعه هنا..."),
])

# كود الرجوع للصفحات
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/app1':
        return app1_layout
    elif pathname == '/app2':
        return app2_layout
    elif pathname == '/app3':
        return app3_layout
    elif pathname == '/app4':
        return app4_layout
    elif pathname == '/app5':
        return app5_layout
    else:
        return "مرحبًا بك في التطبيق الرئيسي! اختر أحد التطبيقات الفرعية من القائمة."

# تشغيل التطبيق
if __name__ == '__main__':
    app.run_server(debug=True)
