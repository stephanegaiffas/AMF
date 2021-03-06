

# bokeh serve --show --port=5007 amf_viz.py &


import numpy as np
import pandas as pd

np.set_printoptions(precision=2)

import os

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from amf.forest import AMFClassifier


from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import Slider, Button
from bokeh.plotting import figure
from bokeh.palettes import RdBu

import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


n_samples = 500
n_features = 2
max_iter = 100
# max_iter = 0


X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=0)


X = MinMaxScaler().fit_transform(X)

n_classes = int(y.max() + 1)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

grid_size = 100
xx, yy = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
xy = np.array([xx.ravel(), yy.ravel()]).T
xy = np.ascontiguousarray(xy, dtype='float32')


def get_tree(of):

    df = of.get_nodes_df(0)

    df['min_x'] = df['memory_range_min'].apply(lambda t: t[0])
    df['min_y'] = df['memory_range_min'].apply(lambda t: t[1])
    df['max_x'] = df['memory_range_max'].apply(lambda t: t[0])
    df['max_y'] = df['memory_range_max'].apply(lambda t: t[1])

    df['count_0'] = df['counts'].apply(lambda t: t[0])
    df['count_1'] = df['counts'].apply(lambda t: t[1])

    df.sort_values(by=['depth', 'parent', 'id'], inplace=True)
    # max_depth = df.depth.max()
    max_depth = 10
    n_nodes = df.shape[0]
    x = np.zeros(n_nodes)
    x[0] = 0.5
    indexes = df['id'].values
    df['x'] = x
    df['y'] = max_depth - df['depth']
    df['x0'] = df['x']
    df['y0'] = df['y']
    for node in range(1, n_nodes):
        index = indexes[node]
        parent = df.at[index, 'parent']
        depth = df.at[index, 'depth']
        left_parent = df.at[parent, 'left']
        x_parent = df.at[parent, 'x']
        if left_parent == index:
            # It's a left node
            df.at[index, 'x'] = x_parent - 0.5 ** (depth + 1)
        else:
            df.at[index, 'x'] = x_parent + 0.5 ** (depth + 1)
        df.at[index, 'x0'] = x_parent
        df.at[index, 'y0'] = df.at[parent, 'y']

    df['color'] = df['is_leaf'].astype('str')
    # df['color'] = df['memorized'].astype('str')
    df.replace({'color': {'False': 'blue', 'True': 'green'}}, inplace=True)
    return df


of = AMFClassifier(n_classes=n_classes, random_state=1234,
                   use_aggregation=True,
                   n_estimators=50, split_pure=True,
                   dirichlet=0.5, step=1.)

dfs = {}
df_datas = {}
zzs = {}
y_color = {0: 'blue', 1: 'red'}


for t in range(0, max_iter + 1):
    x_t = X_train[t].reshape(1, n_features).astype('float32')
    # print('t:', t, ', x_t:', x_t)
    y_t = np.array([y_train[t]]).astype('float32')
    of.partial_fit(x_t, y_t)
    verbose = (t % 10) == 0
    dfs[t] = get_tree(of)
    df_data = pd.DataFrame({'x1': X_train[:(t + 1), 0],
                            'x2': X_train[:(t + 1), 1],
                            'y': y_train[:(t + 1)],
                            't': np.arange(t+1)},
                           columns=['x1', 'x2', 'y', 't'])
    df_data['y'] = df_data['y'].map(lambda y: y_color[y])
    df_datas[t] = df_data

    # print(xy.shape, xy.dtype)
    # print(xy.flags)
    pred = of.predict_proba(xy)

    # zzs[t] = of.predict_proba(xy)[:, 1]
    zzs[t] = of.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)

    if verbose:
        logging.info("Done with step %d" % t)

df = dfs[0]

df_data = df_datas[0]

zz = zzs[0]

source = ColumnDataSource(ColumnDataSource.from_df(df))
source_data = ColumnDataSource(ColumnDataSource.from_df(df_data))
source_decision = ColumnDataSource(data={'image': [zz]})


plot = figure(plot_width=1500, plot_height=500, title="Mondrian Tree",
              x_range=[-0.1, 1.1], y_range=[-1, 15])

plot_data = figure(plot_width=500, plot_height=500, title="Decision and data",
                   x_range=[0, 1], y_range=[0, 1])

plot_data.image('image', source=source_decision, x=0, y=0, dw=1, dh=1,
                palette=RdBu[11])

plot.outline_line_color = None
plot.axis.visible = False
plot.grid.visible = False

plot_data.outline_line_color = None
plot_data.grid.visible = None


circles = plot.circle(x="x", y="y", size=10, fill_color="color", name="circles",
                      fill_alpha=0.4, source=source)

circles_data = plot_data.circle(x="x1", y="x2", size=10,
                                color="y", line_width=1, line_color='black',
                                name="circles",
                                alpha=0.8, source=source_data)


hover = HoverTool(
    renderers=[circles],
    tooltips=[
        ('index', '@id'),
        ('depth', '@depth'),
        ('parent', '@parent'),
        ('left', '@left'),
        ('right', '@right'),
        ('is_leaf', '@is_leaf'),
        # ('time', '@time'),
        # ("feature", "@feature"),
        # ("threshold", "@threshold"),
        ("n_samples", "@n_samples"),
        # ("min_x", "@min_x{0.000}"),
        # ("min_y", "@min_y{0.000}"),
        # ("max_x", "@max_x{0.000}"),
        # ("max_y", "@max_y{0.000}"),
        ('count_0', '@count_0'),
        ('count_1', '@count_1'),
        # ("memorized", "@memorized"),
    ]
)


hover_data = HoverTool(
    renderers=[circles_data],
    tooltips=[
        ("t", "@t"),
        ("x1", "@x1{0.000}"),
        ("x2", "@x2{0.000}")
    ]
)

plot.add_tools(hover)

plot_data.add_tools(hover_data)

plot.text(x="x", y="y", text="id", source=source)

plot.segment(x0="x", y0="y", x1="x0", y1="y0", line_color="#151515",
             line_alpha=0.4, source=source)


def update_plot(attrname, old, new):
    t = iteration_slider.value
    source.data = dfs[t].to_dict('list')
    source_data.data = df_datas[t].to_dict('list')
    source_decision.data = {'image': [zzs[t]]}


iteration_slider = Slider(title="Iteration", value=0, start=0,
                          end=max_iter, step=1, width=1000)

iteration_slider.on_change('value', update_plot)

# button = Button(label='► Play', width=60)
#
#
# callback_id = 0
#
#
# def animate_update():
#     step = iteration_slider.value + 1
#     if step > max_iter:
#         step = 0
#     iteration_slider.value = step
#
#
# def animate():
#     global callback_id
#     if button.label == '► Play':
#         button.label = '❚❚ Pause'
#         callback_id = curdoc().add_periodic_callback(animate_update, 1000)
#     else:
#         button.label = '► Play'
#         curdoc().remove_periodic_callback(callback_id)
#
#
# button.on_click(animate)

# inputs = widgetbox(button, iteration_slider)

inputs = widgetbox(iteration_slider)

layout = column(
    row(plot, width=1500, height=500),
    row(plot_data, inputs, width=500, height=500),
    row(inputs, height=50),
)


curdoc().add_root(layout)

curdoc().title = "Mondrian trees"
