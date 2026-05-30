import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inventory_env import InventoryEnv


def get_discrete_state(state):
	return (int(round(state[0])), int(round(state[1])))


st.title("Smart Inventory Management System")

with open("q_table.pkl", "rb") as f:
	q_table = pickle.load(f)

if st.button("Run 1-Year Simulation"):
	env = InventoryEnv()
	state, _ = env.reset()

	stock_levels = []
	profits = []
	demands = []
	total_profit = 0.0

	terminated = False
	while not terminated:
		discrete_state = get_discrete_state(state)
		if discrete_state in q_table:
			action = int(np.argmax(q_table[discrete_state]))
		else:
			action = 0

		actual_demand = float(env.data.loc[env.current_day, "Daily_Demand"])
		next_state, reward, terminated, truncated, _ = env.step(action)

		stock_levels.append(float(next_state[0]))
		profits.append(float(reward))
		demands.append(actual_demand)
		total_profit += float(reward)

		state = next_state
		if truncated:
			break

	st.metric("Total Annual Profit", f"{total_profit:.2f}")

	stock_fig, stock_ax = plt.subplots()
	stock_ax.plot(stock_levels)
	stock_ax.set_title("Stock Levels Over Time")
	stock_ax.set_xlabel("Day")
	stock_ax.set_ylabel("Stock Level")
	st.pyplot(stock_fig)

	profit_df = pd.DataFrame({"Daily Profit": profits})
	st.line_chart(profit_df)
