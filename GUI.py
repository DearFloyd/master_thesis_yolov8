### gradio-4.3.0 ###
import gradio as gr
import numpy as np
import time
import altair as alt
import pandas as pd
from vega_datasets import data

def make_plot(plot_type):
    if plot_type == "scatter_plot":
        source = pd.read_csv("/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/result_test_01_04_15min.csv")
        # cars = data.cars()
        return alt.Chart(source, title="各行为分布0-15分钟").mark_line().encode(
            # alt.X('timesteps', bin=True),
            x='timesteps',
            y='count',
            color='action',
            # longitude=2,
        ).properties(width=1000, height=500)
        # return alt.Chart(source).mark_line(point=True).encode(
        #     x='timesteps',
        #     y='count',
        #     color='action',
        # )
    elif plot_type == "heatmap":
        # Compute x^2 + y^2 across a 2D grid
        x, y = np.meshgrid(range(-5, 5), range(-5, 5))
        z = x ** 2 + y ** 2

        # Convert this grid to columnar data expected by Altair
        source = pd.DataFrame({'x': x.ravel(),
                            'y': y.ravel(),
                            'z': z.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x='x:O',
            y='y:O',
            color='z:Q'
        )
    elif plot_type == "us_map":
        states = alt.topo_feature(data.us_10m.url, 'states')
        source = data.income.url

        return alt.Chart(source).mark_geoshape().encode(
            shape='geo:G',
            color='pct:Q',
            tooltip=['name:N', 'pct:Q'],
            facet=alt.Facet('group:N', columns=2),
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(data=states, key='id'),
            as_='geo'
        ).properties(
            width=300,
            height=175,
        ).project(
            type='albersUsa'
        )
    elif plot_type == "interactive_barplot":
        source = data.movies.url

        pts = alt.selection(type="single", encodings=['x'])

        rect = alt.Chart(data.movies.url).mark_rect().encode(
            alt.X('IMDB_Rating:Q', bin=True),
            alt.Y('Rotten_Tomatoes_Rating:Q', bin=True),
            alt.Color('count()',
                scale=alt.Scale(scheme='greenblue'),
                legend=alt.Legend(title='Total Records')
            )
        )

        circ = rect.mark_point().encode(
            alt.ColorValue('grey'),
            alt.Size('count()',
                legend=alt.Legend(title='Records in Selection')
            )
        ).transform_filter(
            pts
        )

        bar = alt.Chart(source).mark_bar().encode(
            x='Major_Genre:N',
            y='count()',
            color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
        ).properties(
            width=550,
            height=200
        ).add_selection(pts)

        plot = alt.vconcat(
            rect + circ,
            bar
        ).resolve_legend(
            color="independent",
            size="independent"
        )
        return plot
    elif plot_type == "radial":
        source = pd.DataFrame({"values": [12, 23, 47, 6, 52, 19]})

        base = alt.Chart(source).encode(
            theta=alt.Theta("values:Q", stack=True),
            radius=alt.Radius("values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
            color="values:N",
        )

        c1 = base.mark_arc(innerRadius=20, stroke="#fff")

        c2 = base.mark_text(radiusOffset=10).encode(text="values:Q")

        return c1 + c2

    elif plot_type == "bar_chart":
        source = pd.read_csv("/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/result_test.csv")
        return alt.Chart(source, title="状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1000, height=500)  # 需要记得修改altair库中的/altair/vegalite/data.py中的max_rows: int 从5000到50000


if __name__ == "__main__":

    with gr.Blocks() as demo:
        button = gr.Radio(label="Plot type",
                        choices=['scatter_plot', 'us_map',
                                'interactive_barplot', "radial", 'bar_chart'], value='scatter_plot')
        plot = gr.Plot(label="Plot")
        button.change(make_plot, inputs=button, outputs=[plot])
        demo.load(make_plot, inputs=[button], outputs=[plot])
    
    demo.launch()
