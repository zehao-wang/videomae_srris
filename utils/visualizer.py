import plotly.express as px
import plotly.graph_objects as go
from scipy.special import softmax
import numpy as np
import dash
from dash import dcc, html, Input, Output
import os

def color(color, text):
    return f"<span style='color:{str(color)}'> {str(text)} </span>"

colors = ['red', 'green', 'yellow', 'blue']


class Visualizer(object):
    def __init__(self, mat, sub_folder='', meta=None) -> None:
        """
        Args:
            mat: expect shape of [n_cls, n_clips]
        """
        labels=['Checking_Temperature', 'Cleaning_Plate', 'Closing_Clamps', 'Closing_Doors', 'Opening_Clamps', 'Opening_Doors', 'Putting_Plate_Back', 'Removing_Plate']
        labels=['Checking_Temperature', 'Opening_Doors', 'Opening_Clamps', 'Removing_Plate', 'Cleaning_Plate', 'Putting_Plate_Back', 'Closing_Clamps', 'Closing_Doors']
        # NOTE select color map from colorscales = px.colors.named_colorscales() 
        print(mat.shape, len(labels), len(meta['pred_seq']))
        # x = [
        #     [i for i in range(len(meta['pred_seq']))],
        #     [v for i, v in enumerate(meta['pred_seq'])]
        # ]
        x = [i for i in range(len(meta['pred_seq']))]
        fig = px.imshow(mat, 
                    labels=dict(x="Clip Index", y="Action Type", color="Probability"),
                x = x,
                y=labels,
                text_auto=True,
                color_continuous_scale='viridis',
        )

        tick_texts = []
        for i, (v, gt_l) in enumerate(zip(meta['pred_seq'], meta['gt_seq'])):
            if v == gt_l:
                tick_texts.append(color('green', v))
            else:
                tick_texts.append(color('red', "pred: "+v + "<br>" + "gt: "+gt_l))

        # fig.update_xaxes(tickangle=90)
        fig.update_xaxes(side="top")
        fig.update_layout(
            xaxis = dict(
                tickfont = dict(size=10),
                tickmode = 'array',
                tickvals = [i for i in range(len(meta['pred_seq']))],
                # ticktext = [str(i)+"-"+v for i, v in enumerate(meta['pred_seq'])],
                ticktext = tick_texts,
            )
        )

        app = dash.Dash('app')

        app.layout = html.Div([
            html.H2(id='text1',
                children=[
                    f"Meta info of video {meta['vid']}: ", 
                    
                ]
            ),
            # html.H4(id='text2',
            #     children=[
            #         f"Ground Truth action sequence: ", 
            #         str(meta['gt_seq'])
            #     ]
            # ),
            # html.H4(id='text3',
            #     children=[
            #         f"Predicted action sequence (sliding window): ", 
            #         str(meta['pred_seq'])
            #     ]
            # ),
            dcc.Graph(
                id='heatmap', 
                figure=fig,
            ),
            html.Video(
                controls = True,
                id = 'video',
                src = None,
                autoPlay=True,
                width=400,
                muted = True
            ),
        ])


        @app.callback(
            Output('video', 'src'),
            Input('heatmap', 'hoverData'),
            prevent_initial_call=True
        )
        def update_video(hover_data):
            print(hover_data)
            clip_index = hover_data['points'][0]['x']

            video_file = os.path.join(sub_folder, f"{clip_index}.mp4")
            return video_file
            # if clip_index == 0:
            #     video_file='012_0.mp4'
            # else:
            #     video_file='016_30.mp4'
            # # print( dash.get_asset_url(video_file))
            # return dash.get_asset_url(video_file)
        
        self.app = app

    def run(self, port):
        self.app.run_server( "127.0.0.1", port=port)
