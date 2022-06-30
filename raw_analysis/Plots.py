import json, urllib
import plotly.offline as py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def Sankey0():

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["DFT outputs (480 000)", "Good termination (473 000)", "Bad termination (6525)", "Converged (447 100)", "Error (27 000)"],
            color=['#6940B0','#2E9C9C', '#FF9D4C', '#3DCD3D', '#FF4C4C']
        ),
        link=dict(
            source=[0, 0, 1, 1],  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=[1, 2, 3, 4],
            value=[473000, 6525, 447100, 27000],
            color=['#d5ebeb', '#fff0e4', '#d8f5d8', '#ffdbdb']
        ))])

    fig.update_layout(font_size=22)
    py.plot(fig, validate=False)


def SankeyErrors():

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Terminated with error", "Good termination (473 000)", "Bad termination (6525)", "Converged (447 100)", "Error (27 000)"],
            color=['#6940B0','#2E9C9C', '#FF9D4C', '#3DCD3D', '#FF4C4C']
        ),
        link=dict(
            source=[0, 0, 1, 1],  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=[1, 2, 3, 4],
            value=[473000, 6525, 447100, 27000],
            color=['#d5ebeb', '#fff0e4', '#d8f5d8', '#ffdbdb']
        ))])

    fig.update_layout(font_size=22)
    py.plot(fig, validate=False)



def Sankey():
    scottish_df = pd.read_csv('calcs.csv')

    data_trace = dict(
        type='sankey',
        domain = dict(
          x =  [0,1],
          y =  [0,1]
        ),
        orientation = "h",
        valueformat = ".0f",
        node = dict(
          pad = 10,
          thickness = 30,
          line = dict(
            color = "black",
            width = 0
          ),
          label =  scottish_df['Node, Label'].dropna(axis=0, how='any'),
          color = scottish_df['Color']
        ),
        link = dict(
          source = scottish_df['Source'].dropna(axis=0, how='any'),
          target = scottish_df['Target'].dropna(axis=0, how='any'),
          value = scottish_df['Value'].dropna(axis=0, how='any'),
          color = scottish_df['Link Color'].dropna(axis=0, how='any'),
      )
    )

    layout =  dict(
        title = "Calculation outcomes",
        height = 772,
        width = 950,
        font = dict(
          size = 10
        ),
    )

    fig = dict(data=[data_trace], layout=layout)
    py.plot(fig, validate=False)