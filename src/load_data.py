#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load datasets
"""
import pandas as pd
from typing import Tuple

class DatasetUnga(object):
    """
    Class to handle loading and processing of UNGA voting datasets.
    """

    def __init__(self, path: str = "../dataset/2025_7_21_ga_voting.csv", unsc_path: str = "../2025_7_21_sc_voting.csv") -> None:
        """
        Initialize the DatasetUnga with the path to the dataset.

        Parameters
        ----------
        path : str
            Path to the UNGA voting dataset CSV file.
        """
        self.path = path
        self.unsc_path = unsc_path
        self.load_unga()

    def load_unga(self) -> None:
        """
        Load the UNGA voting dataset.
        """
        df = pd.read_csv(self.path)
        df.columns = [col.strip() for col in df.columns]
        df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True).astype(int)
        self.df = df

    def load_unsc(self) -> None:
        """
        Load the UN Security Council voting dataset.
        """
        if self.unsc_path is None:
            self.unsc_df = None
            return
        df = pd.read_csv(self.unsc_path)
        df.columns = [col.strip() for col in df.columns]
        df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True).astype(int)
        # Match the columns to the UNGA dataset
        if 'permanent_member' in df.columns:
            df = df.drop(columns=['permanent_member', 'modality'])
        df['session'] = pd.NA
        df['committe_report'] = pd.NA
        df = df.rename(columns={
            'description': 'title',
            'agenda': 'agenda_title'
        })
        # Reorder columns to match self.df columns
        df = df.reindex(columns=self.df.columns, fill_value=pd.NA)
        self.unsc_df = df

        # Concatenate the UNGA and UNSC datasets
        self.df = pd.concat([self.df, self.unsc_df], ignore_index=True)
        self.df.reset_index(drop=True, inplace=True)

    def voting_correlation(self, country1: str, country2: str, start_date: str = None, end_date: str = None, abstention_as_no: bool = False) -> float:
        """
        Calculate the correlation of voting patterns between two countries.

        Parameters
        ----------
        country1 : str
            The first country code (e.g., 'USA').
        country2 : str
            The second country code (e.g., 'RUS').
        start_date : str, optional
            Start date for filtering the dataset (format: 'YYYY-MM-DD').
        end_date : str, optional
            End date for filtering the dataset (format: 'YYYY-MM-DD').

        Returns
        -------
        float
            The correlation coefficient between the voting patterns of the two countries.
        """
        if abstention_as_no:
            # If abstentions are treated as 'no', map them to 0
            # This treats abstentions as demonstrating a lack of support
            vote_map = {'Y': 1, 'N': 0, 'A': 0, 'X': 0}
        else:
            # If abstentions are treated as neutral, map them to 0.5
            # This allows for a more nuanced correlation calculation
            vote_map = {'Y': 1, 'N': 0, 'A': 0.5, 'X': 0.5}

        # parse dates
        if start_date is None:
            start_year = min(self.df['year'])
            start_month = min(self.df['month'])
            start_day = min(self.df['day'])
        else:
            start_year, start_month, start_day = (int(x) for x in start_date.split('-'))
        if end_date is None:
            end_year = max(self.df['year'])
            end_month = max(self.df['month'])
            end_day = max(self.df['day'])
        else:
            end_year, end_month, end_day = (int(x) for x in end_date.split('-'))

        # Filter the DataFrame by date if start_date and end_date are provided
        df_filtered = self.df[
            (self.df['year'] > start_year) |
            (self.df['year'] == start_year) & (self.df['month'] > start_month) |
            (self.df['year'] == start_year) & (self.df['month'] == start_month) & (self.df['day'] >= start_day)
        ]
        df_filtered = df_filtered[
            (df_filtered['year'] < end_year) |
            (df_filtered['year'] == end_year) & (df_filtered['month'] < end_month) |
            (df_filtered['year'] == end_year) & (df_filtered['month'] == end_month) & (df_filtered['day'] <= end_day)
        ]

        # Create a pivot table for all countries and votes
        df_pivot = df_filtered.pivot_table(index='ms_code', columns='resolution', values='ms_vote', aggfunc='first')

        if country1 not in df_pivot.index or country2 not in df_pivot.index:
            raise ValueError(f"One or both country codes not found in the dataset: {country1}, {country2}")

        country1_votes = df_pivot.loc[country1].map(vote_map)
        country2_votes = df_pivot.loc[country2].map(vote_map)

        mask = country1_votes.notna() & country2_votes.notna()
        country1_votes = country1_votes[mask]
        country2_votes = country2_votes[mask]

        correlation = country1_votes.corr(country2_votes)
        return correlation

    def correlation_matrix(self, start_date: str = None, end_date: str = None, abstention_as_no: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the correlation matrix of voting patterns for all countries.

        Parameters
        ----------
        start_date : str, optional
            Start date for filtering the dataset (format: 'YYYY-MM-DD').
        end_date : str, optional
            End date for filtering the dataset (format: 'YYYY-MM-DD').
        abstention_as_no : bool, optional
            If True, treat abstentions as 'no' votes.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the correlation matrix of voting patterns.
        """
        if abstention_as_no:
            vote_map = {'Y': 1, 'N': 0, 'A': 0, 'X': 0}
        else:
            vote_map = {'Y': 1, 'N': 0, 'A': 0.5, 'X': 0.5}

        # parse dates
        if start_date is None:
            start_year = min(self.df['year'])
            start_month = min(self.df['month'])
            start_day = min(self.df['day'])
        else:
            start_year, start_month, start_day = (int(x) for x in start_date.split('-'))
        if end_date is None:
            end_year = max(self.df['year'])
            end_month = max(self.df['month'])
            end_day = max(self.df['day'])
        else:
            end_year, end_month, end_day = (int(x) for x in end_date.split('-'))

        # Filter the DataFrame by date if start_date and end_date are provided
        df_filtered = self.df[
            (self.df['year'] > start_year) |
            (self.df['year'] == start_year) & (self.df['month'] > start_month) |
            (self.df['year'] == start_year) & (self.df['month'] == start_month) & (self.df['day'] >= start_day)
        ]
        df_filtered = df_filtered[
            (df_filtered['year'] < end_year) |
            (df_filtered['year'] == end_year) & (df_filtered['month'] < end_month) |
            (df_filtered['year'] == end_year) & (df_filtered['month'] == end_month) & (df_filtered['day'] <= end_day)
        ]

        # Create a pivot table for all countries and votes
        df_pivot = df_filtered.pivot_table(index='ms_code', columns='resolution', values='ms_vote', aggfunc='first')

        # Remove rows and columns where all votes are 'X'
        rows_all_x = df_pivot.apply(lambda row: all(v in ['A', 'X'] for v in row), axis=1)
        cols_all_x = df_pivot.apply(lambda col: all(v in ['A', 'X'] for v in col), axis=0)
        df_pivot = df_pivot.loc[~rows_all_x, ~cols_all_x]

        # Map votes to numeric values
        df_pivot = df_pivot.map(vote_map.get)

        notna_mask = {}
        for country in df_pivot.index:
            notna_mask[country] = df_pivot.loc[country].notna()

        # Calculate the upper triangle of correlation matrix
        corr_matrix = pd.DataFrame(index=df_pivot.index, columns=df_pivot.index)
        for country1 in df_pivot.index:
            for country2 in df_pivot.index[df_pivot.index.get_loc(country1):]:  # Avoid duplicate calculations
                if country1 == country2:
                    corr_matrix.loc[country1, country2] = 1.0
                    continue

                # Get the votes for both countries
                country1_votes = df_pivot.loc[country1]
                country2_votes = df_pivot.loc[country2]

                # Filter out NaN votes
                mask = notna_mask[country1] & notna_mask[country2]
                country1_votes = country1_votes[mask]
                country2_votes = country2_votes[mask]

                if len(country1_votes) < 2:
                    # Not enough data to calculate correlation, perform a direct comparison
                    corr_matrix.loc[country1, country2] = 1.0 if country1_votes.equals(country2_votes) else 0.0
                elif country1_votes.equals(country2_votes):
                    # If both countries have the same votes, set correlation to 1
                    corr_matrix.loc[country1, country2] = 1.0
                elif country1_votes.std() == 0 or country2_votes.std() == 0:
                    # If either country's votes have zero standard deviation, correlation is undefined but set to 0
                    corr_matrix.loc[country1, country2] = 0.0
                else:
                    # Calculate the correlation between the two countries' votes
                    corr_matrix.loc[country1, country2] = country1_votes.corr(country2_votes)
                    # Calculate the correlation between the two countries' votes
                    corr_matrix.loc[country1, country2] = country1_votes.corr(country2_votes)

        # If a column is all NaN, drop it
        corr_matrix.dropna(axis=0, how='all', inplace=True)
        corr_matrix.dropna(axis=1, how='all', inplace=True)

        # Fill the lower triangle of the matrix
        for country1 in corr_matrix.index:
            for country2 in corr_matrix.index:
                if country1 != country2 and pd.isna(corr_matrix.loc[country2, country1]):
                    corr_matrix.loc[country2, country1] = corr_matrix.loc[country1, country2]

        names = self.df[['ms_code', 'ms_name']].drop_duplicates().set_index('ms_code')
        names = names.groupby('ms_code')['ms_name'].apply(lambda x: '; '.join(sorted(set(x)))).to_frame()
        names = names.loc[corr_matrix.index]

        return corr_matrix, names

def load_unga(path: str = "../dataset/2025_7_21_ga_voting.csv") -> pd.DataFrame:
    """
    Load the UNGA voting dataset.

    Parameters
    ----------
    path : str
        Path to the UNGA voting dataset CSV file.
    """
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True).astype(int)
    return df

if __name__ == "__main__":
    import time

    # time benchmark
    ds = DatasetUnga(path="../dataset/2025_7_23_ga_voting.csv", unsc_path="../dataset/2025_7_23_sc_voting.csv")
    start_time = time.time()
    corr_matrix, names = ds.correlation_matrix(start_date='2024-01-01', end_date='2024-12-31')
    end_time = time.time()
    print(f"Time taken to compute correlation matrix: {end_time - start_time} seconds")