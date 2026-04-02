"""Layout components."""

import streamlit as st


def create_two_column_layout(left_ratio: int = 1, right_ratio: int = 1) -> tuple:
    """Create a two-column layout."""
    return st.columns([left_ratio, right_ratio])


def create_three_column_layout(ratios: tuple = (1, 1, 1)) -> tuple:
    """Create a three-column layout."""
    return st.columns(ratios)
