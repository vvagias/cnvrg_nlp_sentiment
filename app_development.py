import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import http.client
import json
import ssl

#dash app
app = dash.Dash(__name__)

app.layout = html.Div([
html.H1(children="Sentiment Analysis Application"),
html.H2(children="huggingface / pytorch / dash / cnvrg.io"),
html.Tr([
html.Br(),
"Evaluate Sentiment!",

html.Div(id='my-confirmation'),
html.Div([dcc.Input(id='my-input', value='This is Amazing!', type='text'),
html.Button(id='submit-button-state', n_clicks=0, children='Submit')]),
html.Br(),

html.H6('https://cnvrg.io/developers'),
]),
html.H2(id='my-answer', children="response")
])


@app.callback(
    Output(component_id='my-answer', component_property='children'),
    Input(component_id='submit-button-state', component_property='n_clicks'),
    State(component_id='my-input', component_property='value')

)
def update_output_div(n_clicks, input_value):
    # loop on user's or prepared questions
    #generate response
    import http.client
    conn = http.client.HTTPSConnection("sentiment-ws-1.aalk4m89gwugsbjwmhe9fg6.cloud.cnvrg.io")
    #conn = http.client.HTTPSConnection("sentiment-1.prod.cnvrg.io")
    payload = "{\"input_params\": \"" + input_value + "\"}"
    headers = {
        'Cnvrg-Api-Key': "DGTCjezhqgHY38ZGKQAfphNW",
        'Content-Type': "application/json"
        }

    conn.request("POST", "/api/v1/endpoints/myvlfvccztf9deaaxc4y", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = json.loads(data)
    
    return str(json.dumps(data))




#server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=6006) #run in tenorboard session tab
