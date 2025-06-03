#!/usr/bin/env python
# coding: utf-8


#import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

#page 1 layout
st.set_page_config(page_title="Sales-Ninja Dashboard", layout="wide")
st.title ("Sales-Ninja Dashboard")
#tabs optional
tab1, tab2 = st.tabs(['tab1', 'tab2'])

with tab1:
    st.markdown("## Overview")

with tab2:
    st.subheader("Coming optionally")


# --- Load data ---
