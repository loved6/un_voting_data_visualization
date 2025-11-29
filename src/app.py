#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotly Dash application for visualizing UN voting data."""
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import glob
import os
from pycountry_convert import country_alpha2_to_continent_code, convert_continent_code_to_continent_name
from sklearn.cluster import KMeans, OPTICS, DBSCAN, HDBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from typing import Tuple, List
import dash_bootstrap_components as dbc
from load_data import DatasetUnga

def get_corr_matrix(ds: DatasetUnga, start_date: str, end_date: str, abstention_as_no: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the correlation matrix for the dataset.

    Parameters
    ----------
    ds : DatasetUnga
        UNGA voting dataset handler.

    Returns
    -------
    pd.DataFrame
        Correlation matrix of the dataset.
    """
    return ds.correlation_matrix(start_date=start_date, end_date=end_date, abstention_as_no=abstention_as_no)

def get_pca_data(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute the PCA data for the dataset.

    Parameters
    ----------
    ds : DatasetUnga
        UNGA voting dataset handler.

    Returns
    -------
    pd.DataFrame
        PCA-transformed dataset.
    """
    pca = PCA(n_components=2)
    corr_matrix_pca = pca.fit_transform(corr_matrix)
    return pd.DataFrame(corr_matrix_pca, index=corr_matrix.index, columns=['PC1', 'PC2'])

def cluster_data(corr_matrix: pd.DataFrame, method: str = 'kmeans', **kwargs) -> pd.DataFrame:
    """Cluster the correlation matrix data using the specified method.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix of the dataset.
    method : str
        Clustering method to use ('kmeans', 'optics', 'dbscan', 'hdbscan', 'affinity_propagation').

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster labels.
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=kwargs.get('n_clusters', 5))
    elif method == 'optics':
        model = OPTICS(min_samples=kwargs.get('min_samples', 3))
    elif method == 'dbscan':
        model = DBSCAN(eps=kwargs.get('eps', 1.5), min_samples=kwargs.get('min_samples', 3))
    elif method == 'hdbscan':
        model = HDBSCAN(min_cluster_size=kwargs.get('min_cluster_size', 3))
    elif method == 'affinity_propagation':
        model = AffinityPropagation()
    else:
        raise ValueError("Unsupported clustering method.")
    labels = model.fit_predict(corr_matrix)
    return pd.DataFrame(labels, index=corr_matrix.index, columns=['cluster'])

def gen_scatter_plot(corr_matrix_pca: pd.DataFrame, cluster_labels: pd.DataFrame, names:pd.DataFrame, highlight: List[List[str]]=[], highlight_label_only: bool=False) -> go.Figure:
    """Generate a scatter plot of the PCA data with cluster labels.

    Parameters
    ----------
    corr_matrix_pca : pd.DataFrame
        PCA-transformed correlation matrix.
    cluster_labels : pd.DataFrame
        DataFrame containing cluster labels.
    names : pd.DataFrame
        DataFrame containing country names.
    highlight : List[List[str]]
        List of countries to highlight in the scatter plot with each sublist assigned a different outline color.
    highlight_label_only : bool
        If True, only the labels of highlighted countries will be shown without markers.

    Returns
    -------
    go.Figure
        Plotly figure object for the scatter plot.
    """
    # Set default opacity values
    nonfocused_opacity = 0.3
    focused_opacity = 1.0

    # Prepare dataframe
    scatter_df = corr_matrix_pca.copy()
    scatter_df['cluster'] = cluster_labels['cluster']
    scatter_df['ms_name'] = scatter_df.index.map(lambda x: names.loc[x, 'ms_name'] if x in names.index else '')
    scatter_df['category'] = [f"Cluster {i}" for i in scatter_df['cluster'].astype(str)]
    nclusters = scatter_df['category'].nunique()

    if scatter_df.isnull().values.any():
        raise ValueError("The PCA DataFrame contains NaN values. Please check the input data.")

    # Build cluster colored scatter plot
    if highlight_label_only:
        opacity = 1.0
    else:
        opacity=1.0 if len(highlight) == 0 else nonfocused_opacity
    fig = px.scatter(
        scatter_df,
        x='PC1',
        y='PC2',
        text=None,
        color='category',
        opacity=opacity,
        title="PCA Scatterplot of UN Members' Voting Data",
        hover_name=scatter_df['ms_name'],
        labels={'category': 'Cluster'},
    )

    if len(highlight) > 0:
        # Add highlighted codes with higher opacity
        for group in highlight:
            highlighted_df = scatter_df.loc[group]
            # Assign a distinct color for each group using Plotly's qualitative palette
            colors = px.colors.qualitative.Plotly
            color = colors[(highlight.index(group) + nclusters) % len(colors)]
            name = f'{", ".join(group)}'
            if len(name) > 12:
                name = name[:12] + '...'
            if highlight_label_only:
                color = 'rgba(0, 0, 0, 0)'  # Transparent marker for labels only
            fig.add_trace(
                go.Scatter(
                    x=highlighted_df['PC1'],
                    y=highlighted_df['PC2'],
                    mode='markers+text',
                    marker=dict(
                        color=color,
                        line=dict(width=2, color='black')
                    ),
                    text=highlighted_df.index,
                    textposition='top center',
                    name=name,
                    hovertext=[
                        f"{highlighted_df.loc[code, 'ms_name']}<br>ms_code: {code}<br>Cluster: {highlighted_df.loc[code, 'cluster']}"
                        for code in highlighted_df.index
                    ],
                    opacity=focused_opacity,
                    showlegend=True
                )
            )

    fig.update_layout(showlegend=True)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(textposition='top center')
    fig.update_xaxes(title_text='Principal Component 1')
    fig.update_yaxes(title_text='Principal Component 2')
    return fig

def gen_treemap(names: pd.DataFrame, cluster_labels: pd.DataFrame) -> go.Figure:
    """Generate a treemap of the correlation matrix.

    Parameters
    ----------
    names : pd.DataFrame
        DataFrame containing country names.
    cluster_labels : pd.DataFrame
        DataFrame containing cluster labels.

    Returns
    -------
    go.Figure
        Plotly figure object for the treemap.
    """
    # Find continents for each country in the correlation matrix
    continents = ['']*len(cluster_labels)
    for i, code in enumerate(cluster_labels.index):
        country = pycountry.countries.get(alpha_3=code)
        try:
            if country is None:
                continents[i] = 'Unknown'
            else:
                continents[i] = convert_continent_code_to_continent_name(country_alpha2_to_continent_code(country.alpha_2))
        except KeyError:
            continents[i] = 'Unknown'

    treemap_df = cluster_labels.copy()
    treemap_df['ms_code'] = treemap_df.index
    treemap_df['continent'] = continents
    treemap_df['ms_name'] = treemap_df.index.map(lambda x: names.loc[x, 'ms_name'] if x in names.index else '')

    fig = px.treemap(
        treemap_df,
        path=['cluster', 'continent', 'ms_name'],
        values=None,
        title='Treemap of Country Clusters',
        hover_data=['ms_name', 'continent', 'ms_code'],
    )
    return fig

def get_statistics_table(corr_matrix: pd.DataFrame, cluster_labels: pd.DataFrame, highlight: List[List[str]]=[]) -> dict:
    """Generate a statistics table for the country groups/clusters.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        DataFrame containing correlation matrix.
    cluster_labels : pd.DataFrame
        DataFrame containing cluster labels.
    highlight : list
        List of country codes to highlight.

    Returns
    -------
    Dict
        Dictionary containing the statistics table data.
    """
    groups = {}
    if len(highlight) > 0:
        for group in highlight:
            # Ensure the group is a list of country codes
            if isinstance(group, str):
                group = [code.strip() for code in group.split(',')]
            # Create a name for the group
            name = f'{", ".join(group)}'
            if len(name) > 12:
                name = name[:12] + '...'
            groups[name] = group

    stats_df = corr_matrix.copy()
    stats_df['cluster'] = cluster_labels['cluster']
    stats_df['group'] = stats_df.index.map(lambda x: next((name for name, group in groups.items() if x in group), None))

    groups = {}
    clusters = stats_df['cluster'].unique()
    clusters = sorted(clusters)
    for cluster in clusters:
        subset = corr_matrix[stats_df['cluster'] == cluster]
        if subset.shape[0] > 1:
            # Standard deviation in Euclidean space of all samples
            values = subset.values
            centroid = values.mean(axis=0)
            distances_from_centroid = ((values - centroid) ** 2).sum(axis=1) ** 0.5
            dispersion = np.mean(distances_from_centroid)

            groups[f"Cluster {cluster}"] = {
                'Dispersion': f"{dispersion:.3f}",
                'Size': subset.shape[0],
            }
    for group in stats_df['group'].unique():
        if group is not None:
            subset = corr_matrix[stats_df['group'] == group]
            if subset.shape[0] > 1:
                # Standard deviation in Euclidean space of all samples
                values = subset.values
                centroid = values.mean(axis=0)
                distances_from_centroid = ((values - centroid) ** 2).sum(axis=1) ** 0.5
                dispersion = np.mean(distances_from_centroid)

                groups[group] = {
                    'Dispersion': f"{dispersion:.3f}",
                    'Size': subset.shape[0],
                }


    # Create a summary DataFrame
    summary_df = pd.DataFrame.from_dict(groups, orient='index')
    summary_df.reset_index(inplace=True)

    return summary_df

class Visualization(object):
    """Class for visualizing UNGA voting data."""

    def __init__(self, ds: DatasetUnga):
        """Initialize the Visualization class with a dataset."""
        self.ds = ds

    def filter_by_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter the dataset by date range.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing voting data within the specified date range.
        """
        # Get the correlation matrix
        self.corr_matrix, self.names = get_corr_matrix(self.ds, start_date, end_date)

        # Perform PCA
        self.corr_matrix_pca = get_pca_data(self.corr_matrix)

        # Cluster the data
        self.cluster_labels = cluster_data(self.corr_matrix, method='kmeans', n_clusters=2)

    def update_n_clusters(self, n_clusters: int) -> None:
        """Update the number of clusters for KMeans clustering.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to use in KMeans clustering.
        """
        if hasattr(self, 'corr_matrix'):
            self.cluster_labels = cluster_data(self.corr_matrix, method='kmeans', n_clusters=n_clusters)
        else:
            raise ValueError("Correlation matrix not available. Please filter by date first.")

    def gen_scatter_plot(self, highlight_countries: List[List[str]]=[], highlight_label_only: bool=False) -> go.Figure:
        """Generate a scatter plot of the PCA results.

        Parameters
        ----------
        highlight_countries : list
            List of country codes to highlight.
        highlight_label_only : bool
            If True, only the labels of highlighted countries will be shown without markers.

        Returns
        -------
        go.Figure
            Plotly figure object for the scatter plot.
        """
        if not hasattr(self, 'corr_matrix_pca') or not hasattr(self, 'cluster_labels') or not hasattr(self, 'names'):
            return px.scatter(title="No data available. Please filter by date first.")
        else:
            return gen_scatter_plot(self.corr_matrix_pca, self.cluster_labels, self.names, highlight_countries, highlight_label_only)

    def gen_treemap(self) -> go.Figure:
        """Generate a treemap of the clustered data.

        Returns
        -------
        go.Figure
            Plotly figure object for the treemap.
        """
        return gen_treemap(self.names, self.cluster_labels)

    def get_statistics_table(self, highlight: List[List[str]]=[]) -> dict:
        """Generate a statistics table for the country groups/clusters.

        Parameters
        ----------
        highlight : list
            List of country codes to highlight.

        Returns
        -------
        go.Figure
            Plotly figure object for the statistics table.
        """
        if not hasattr(self, 'corr_matrix') or not hasattr(self, 'cluster_labels'):
            return {}
        return get_statistics_table(self.corr_matrix, self.cluster_labels, highlight)

def main():
    """Run the Dash application."""
    app = dash.Dash(__name__)
    app.title = "UN Voting Data Visualization"

    # Load the dataset
    ds = DatasetUnga()

    # Initialize the visualization classes
    viz1 = Visualization(ds)
    viz2 = Visualization(ds)

    # Initialize highlight filters
    highlights = []

    # Set up the layout and callbacks here
    app.layout = dash.html.Div([
        dash.html.Div([
            dash.html.H1("UN Voting Data Visualization", style={'textAlign': 'center'}),
            dash.html.Div([
                dash.html.H2("Time Window 1"),
                dash.html.Div([
                    dash.html.Label("Start Date:"),
                    dash.dcc.Input(id='start-date', type='text', value='2023-01-01'),
                    dash.html.Label("End Date:"),
                    dash.dcc.Input(id='end-date', type='text', value='2023-12-31'),
                    dash.html.Button('Update', id='update-button')
                ]),
                dash.dcc.Graph(id='scatter-plot'),
                dash.dcc.Graph(id='treemap'),
                dbc.Table(
                    id='statistics-table'
                ),
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            dash.html.Div([
                dash.html.H2("Time Window 2"),
                dash.html.Div([
                    dash.html.Label("Start Date:"),
                    dash.dcc.Input(id='start-date-2', type='text', value='2024-01-01'),
                    dash.html.Label("End Date:"),
                    dash.dcc.Input(id='end-date-2', type='text', value='2024-12-31'),
                    dash.html.Button('Update', id='update-button-2')
                ]),
                dash.dcc.Graph(id='scatter-plot-2'),
                dash.dcc.Graph(id='treemap-2'),
                dbc.Table(
                    id='statistics-table-2'
                ),
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            dash.html.Div([
                dash.html.H2("Controls"),
                dash.html.Div([
                    dash.html.Label("Country Groups to Highlight:"),
                    dash.dcc.Input(id='highlight-countries', type='text', value='USA, CHN, RUS'),
                    dash.html.Button('Apply Highlights', id='update-highlights-button'),
                    dash.html.Button('Disable Highlights', id='disable-highlights-button'),
                    dash.html.Button('Group Markers as Cluster Color', id='no-highlight-markers-button'),
                    dash.html.Div(style={'display': 'inline-block', 'width': '16px'}),
                    dash.html.Label("Number of Clusters:"),
                    dash.dcc.Input(id='num-clusters', type='number', value=2, min=1, step=1)
                ], style={'marginBottom': '16px', 'verticalAlign': 'bottom'}),
            ], style={'width': '100%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '100%', 'height': '100%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'width': '100%', 'height': '100%', 'display': 'flex'})

    @app.callback(
        dash.dependencies.Output('scatter-plot', 'figure', allow_duplicate=True),
        dash.dependencies.Output('treemap', 'figure', allow_duplicate=True),
        dash.dependencies.Output('statistics-table', 'children', allow_duplicate=True),
        [dash.dependencies.Input('update-button', 'n_clicks')],
        [dash.dependencies.State('start-date', 'value'),
         dash.dependencies.State('end-date', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def update_graph(n_clicks, start_date, end_date):
        """Update the scatter plot based on user input."""
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Filter the dataset by date range
        viz1.filter_by_date(start_date, end_date)

        # Generate the scatter plot
        fig = viz1.gen_scatter_plot(highlight_countries=highlights)

        # Generate the treemap
        treemap_fig = viz1.gen_treemap()

        stats_df = viz1.get_statistics_table(highlights)

        return fig, treemap_fig, dbc.Table.from_dataframe(stats_df)

    @app.callback(
        dash.dependencies.Output('scatter-plot-2', 'figure', allow_duplicate=True),
        dash.dependencies.Output('treemap-2', 'figure', allow_duplicate=True),
        dash.dependencies.Output('statistics-table-2', 'children', allow_duplicate=True),
        [dash.dependencies.Input('update-button-2', 'n_clicks')],
        [dash.dependencies.State('start-date-2', 'value'),
         dash.dependencies.State('end-date-2', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def update_graph_2(n_clicks, start_date, end_date):
        """Update the second scatter plot based on user input."""
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Filter the dataset by date range
        viz2.filter_by_date(start_date, end_date)

        # Generate the scatter plot
        fig = viz2.gen_scatter_plot(highlight_countries=highlights)

        # Generate the treemap
        treemap_fig = viz2.gen_treemap()

        stats_df = viz2.get_statistics_table(highlights)

        return fig, treemap_fig, dbc.Table.from_dataframe(stats_df)

    @app.callback(
        dash.dependencies.Output('scatter-plot', 'figure', allow_duplicate=True),
        dash.dependencies.Output('scatter-plot-2', 'figure', allow_duplicate=True),
        dash.dependencies.Output('treemap', 'figure', allow_duplicate=True),
        dash.dependencies.Output('treemap-2', 'figure', allow_duplicate=True),
        dash.dependencies.Output('statistics-table', 'children', allow_duplicate=True),
        dash.dependencies.Output('statistics-table-2', 'children', allow_duplicate=True),
        [dash.dependencies.Input('num-clusters', 'value')],
        prevent_initial_call=True
    )
    def update_num_clusters(n_clusters):
        """Update the number of clusters for both scatter plots."""
        if n_clusters is None or n_clusters < 1:
            raise dash.exceptions.PreventUpdate

        # Update the cluster labels for both visualizations
        viz1.update_n_clusters(n_clusters)
        viz2.update_n_clusters(n_clusters)

        # Generate the scatter plots
        fig1 = viz1.gen_scatter_plot(highlight_countries=highlights)
        fig2 = viz2.gen_scatter_plot(highlight_countries=highlights)

        # Generate the treemaps
        treemap_fig1 = viz1.gen_treemap()
        treemap_fig2 = viz2.gen_treemap()

        # Update the statistics tables
        stats_df1 = viz1.get_statistics_table(highlights)
        stats_df2 = viz2.get_statistics_table(highlights)

        return fig1, fig2, treemap_fig1, treemap_fig2, dbc.Table.from_dataframe(stats_df1), dbc.Table.from_dataframe(stats_df2)

    @app.callback(
        dash.dependencies.Output('scatter-plot', 'figure', allow_duplicate=True),
        dash.dependencies.Output('scatter-plot-2', 'figure', allow_duplicate=True),
        dash.dependencies.Output('statistics-table', 'children', allow_duplicate=True),
        dash.dependencies.Output('statistics-table-2', 'children', allow_duplicate=True),
        [dash.dependencies.Input('update-highlights-button', 'n_clicks')],
        [dash.dependencies.State('highlight-countries', 'value')],
        prevent_initial_call=True
    )
    def update_highlights(n_clicks, highlight_countries):
        """Update the highlights on both scatter plots based on user input."""
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Parse the highlight countries input
        highlights = [[country.strip() for country in clist.strip().split(',')] for clist in highlight_countries.split(';')]

        # Update the scatter plots with the new highlights
        fig1 = viz1.gen_scatter_plot(highlight_countries=highlights)
        fig2 = viz2.gen_scatter_plot(highlight_countries=highlights)

        # Update the statistics tables
        stats_df1 = viz1.get_statistics_table(highlights)
        stats_df2 = viz2.get_statistics_table(highlights)

        return fig1, fig2, dbc.Table.from_dataframe(stats_df1), dbc.Table.from_dataframe(stats_df2)

    @app.callback(
        [dash.dependencies.Output('scatter-plot', 'figure', allow_duplicate=True),
         dash.dependencies.Output('scatter-plot-2', 'figure', allow_duplicate=True),
         dash.dependencies.Output('statistics-table', 'children', allow_duplicate=True),
         dash.dependencies.Output('statistics-table-2', 'children', allow_duplicate=True)],
        [dash.dependencies.Input('disable-highlights-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def disable_highlights(n_clicks):
        """Disable highlights on both scatter plots."""
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Update the scatter plots without highlights
        fig1 = viz1.gen_scatter_plot()
        fig2 = viz2.gen_scatter_plot()

        # Update the statistics tables without highlights
        stats_df1 = viz1.get_statistics_table()
        stats_df2 = viz2.get_statistics_table()

        return fig1, fig2, dbc.Table.from_dataframe(stats_df1), dbc.Table.from_dataframe(stats_df2)

    @app.callback(
        [dash.dependencies.Output('scatter-plot', 'figure', allow_duplicate=True),
         dash.dependencies.Output('scatter-plot-2', 'figure', allow_duplicate=True)],
        [dash.dependencies.Input('no-highlight-markers-button', 'n_clicks')],
        [dash.dependencies.State('highlight-countries', 'value')],
        prevent_initial_call=True
    )
    def no_highlight_markers(n_clicks, highlight_countries):
        """Disable highlight markers on both scatter plots."""
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Parse the highlight countries input
        highlights = [[country.strip() for country in clist.strip().split(',')] for clist in highlight_countries.split(';')]

        # Update the scatter plots with no highlight markers
        fig1 = viz1.gen_scatter_plot(highlight_countries=highlights, highlight_label_only=True)
        fig2 = viz2.gen_scatter_plot(highlight_countries=highlights, highlight_label_only=True)

        return fig1, fig2

    # Run the app
    app.run(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()