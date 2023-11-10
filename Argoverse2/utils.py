from config import * 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import plotly.graph_objects as go
# =============================================================================

def normalize(vector, point): 
  return vector - point


def draw_circle(fig, ax, center, radius): 

  circle = plt.Circle(center, radius, edgecolor='k', facecolor='none')

  ax.add_patch(circle)

  ax.set_aspect('equal', adjustable='box')


  return fig, ax

def plot_map(avm, t_ids, df):
    fig, ax = plt.subplots(figsize=(20, 12))
    lane_ids = avm.get_scenario_lane_segment_ids()

    for lane_id in lane_ids:
        lane_polygon = avm.get_lane_segment_polygon(lane_id)
        ax.plot(lane_polygon[:, 0], lane_polygon[:, 1], color='b')

    for track_id in t_ids:
        df_ = df.loc[df['track_id'] == track_id]
        ax.scatter(df_['position_x'], df_['position_y'], color=object_color_code[df_['object_type'].iloc[0]])

    ax.set_xticks([])
    ax.set_yticks([])

    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in object_color_code.values()]
    ax.legend(markers, object_color_code.keys(), numpoints=1, loc='upper left')

    return fig, ax


def plot_track(df: pd.DataFrame, track_id: str, colormap='RdBu'):

  df_ = df.loc[df['track_id']==track_id]
  df_.sort_values(by='timestep', ascending=False)
  x = df_['position_x']
  y = df_['position_y']
  velocity = np.linalg.norm(df_[['velocity_x', 'velocity_y']], axis=1)
  displacement = np.linalg.norm(np.abs(df_[["position_x", "position_y"]].iloc[-1] - df_[["position_x", "position_y"]].iloc[0]))
  trace = go.Scatter(x=x, y=y,
                     mode='markers',
                     marker=dict(
                     size=10,
                     color=-velocity,
                     colorscale=colormap,
                     colorbar=dict(title='Velocity')),
                     hovertext=[f"v: {v:0.5f}<br>t: {str(t)}" for v, t in zip(velocity, df_["timestep"].values)]
                     )
  layout = go.Layout(
      title=f'''
      Trajectory of Object: <b>{track_id}</b>
      <br>Type of Object: <b>{df_["object_type"].iloc[0]}</b>
      &nbsp;Object Category: <b>{track_category_mapping[df_["object_category"].iloc[0]]}</b>
      <br># Frames: <b>{len(df_)}</b>
      &nbsp;Avg. Velocity: <b>{np.mean(velocity):0.5f}</b>
      &nbsp;Total. Displacement: <b>{displacement:0.5f}</b>
      &nbsp;mean. Heading: <b>{np.mean(df_['heading']):0.5f}</b>
      ''',
      xaxis=dict(showgrid=False, showticklabels=False,
                 mirror=True,
                 linecolor='white',
                 tickfont=dict(color='white'),
                 ),
      yaxis=dict(showgrid=False, showticklabels=False,
                 mirror=True,
                 linecolor='white',
                 tickfont=dict(color='white'),
                 ),
      paper_bgcolor='black',
      plot_bgcolor='black',
      font=dict(color='white')
  )

  fig = go.Figure(data=[trace], layout=layout)
  fig.show()