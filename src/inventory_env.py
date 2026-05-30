from pathlib import Path

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd


class InventoryEnv(gym.Env):
	def __init__(self) -> None:
		base_dir = Path(__file__).resolve().parents[1]
		data_path = base_dir / "data" / "inventory_data.csv"
		model_path = base_dir / "models" / "demand_model.joblib"

		self.data = pd.read_csv(data_path)
		self.model = joblib.load(model_path)
		self.features = ["Day_of_Week", "Is_Weekend", "Has_Promotion", "Price"]

		self.action_space = gym.spaces.Discrete(5)
		self.observation_space = gym.spaces.Box(
			low=np.array([0, 0], dtype=np.float32),
			high=np.array([100, 100], dtype=np.float32),
			dtype=np.float32,
		)
		self.action_mapping = np.array([0, 10, 20, 30, 40], dtype=np.int32)

		self.current_day = 0
		self.current_stock = 20

	def _predict_demand(self, day_index: int) -> float:
		feature_row = self.data.loc[[day_index], self.features]
		return float(self.model.predict(feature_row)[0])

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		super().reset(seed=seed)

		self.current_day = 0
		self.current_stock = 20

		predicted_demand = self._predict_demand(self.current_day)
		state = np.array([self.current_stock, predicted_demand], dtype=np.float32)

		return state, {}

	def step(self, action: int):
		order_quantity = int(self.action_mapping[int(action)])
		self.current_stock += order_quantity

		actual_demand = float(self.data.loc[self.current_day, "Daily_Demand"])
		stock_before_sales = float(self.current_stock)

		sold_items = min(stock_before_sales, actual_demand)
		missed_sales = max(0.0, actual_demand - stock_before_sales)

		self.current_stock = stock_before_sales - sold_items
		if self.current_stock > 100:
			self.current_stock = 100

		reward = (sold_items * 10) - (self.current_stock * 2) - (missed_sales * 5)

		self.current_day += 1
		terminated = self.current_day >= 364
		truncated = False

		next_day_index = min(self.current_day, len(self.data) - 1)
		predicted_demand = self._predict_demand(next_day_index)
		state = np.array([self.current_stock, predicted_demand], dtype=np.float32)

		return state, float(reward), terminated, truncated, {}


if __name__ == "__main__":
	env = InventoryEnv()
	state, _ = env.reset()
	next_state, reward, terminated, truncated, info = env.step(
		env.action_space.sample()
	)
	print("State:", next_state)
	print("Reward:", reward)
