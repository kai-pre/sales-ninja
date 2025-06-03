#!/usr/bin/env python
# coding: utf-8


#import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

#get data
fact = pd.read_csv("raw_data/FactSales.csv")
dim_date = pd.read_csv("raw_data/DimDate.csv")
data = fact.merge(dim_date, how="left", on="DateKey")
data["DateKey"] = pd.to_datetime(data["DateKey"])


#page 1 layout
st.set_page_config(page_title="Sales-Ninja Dashboard", layout="wide")
st.title ("Sales-Ninja Dashboard")

st.sidebar.header("Filter Data")
min_date = data["DateKey"].min()
max_date = data["DateKey"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if isinstance(date_range, list) and len(date_range) == 2:
    data = data[(data["DateKey"] >= date_range[0]) & (data["DateKey"] <= date_range[1])]

years = sorted(data["DateKey"].dt.year.unique())
selected_years = st.sidebar.multiselect("Select Year(s)", options=years, default=years)
data = data[data["DateKey"].dt.year.isin(selected_years)]

selected_product = st.sidebar.selectbox("Product Hierarchy", ["All", "Category A", "Category B"])
selected_promotion = st.sidebar.selectbox("Promotion", ["All", "Promo 1", "Promo 2"])
selected_geo = st.sidebar.selectbox("Geography", ["All", "Region A", "Region B"])
#data["Month"] = data["DateKey"].dt.month_name()
#data["Day"] = data["DateKey"].dt.day_name()
#data["Quarter"] = data["DateKey"].dt.quarter


day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
quarter_order = ["Q1", "Q2", "Q3", "Q4"]

# Create the 3 tabs
tab1, tab2, tab3 = st.tabs(["By Day", "By Month", "By Quarter"])

# Tab 1: Day View
with tab1:
    st.subheader("Sales by Day of the Week")

    # KPIs
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col1.metric("Revenue", "$123K","+7.5%")
    col2.metric("Net Sales", "$98K","-3.2%")
    col3.metric("Total Cost", "$76K", "7.5%")
    col4.metric("Sales Value", "$145K","3.5%")
# Tab 2: Month View
with tab2:
    st.subheader("Sales by Month")
    col5, col6, col7, col8 = st.columns([1, 1, 1, 1])
    col5.metric("Revenue", "$123K")
    col6.metric("Net Sales", "$98K")
    col7.metric("Total Cost", "$76K")
    col8.metric("Sales Value", "$145K")
# Tab 3: Quarter View
with tab3:
    st.subheader("Sales by Quarter")
    col9, col10, col11, col12 = st.columns([1, 1, 1, 1])
    col9.metric("Revenue", "$123K")
    col10.metric("Net Sales", "$98K")
    col11.metric("Total Cost", "$76K")
    col12.metric("Sales Value", "$145K")



# --- Load data ---
