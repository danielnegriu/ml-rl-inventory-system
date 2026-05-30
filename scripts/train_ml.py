from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main() -> None:
	base_dir = Path(__file__).resolve().parents[1]
	data_path = base_dir / "data" / "inventory_data.csv"
	models_dir = base_dir / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	data = pd.read_csv(data_path)

	features = ["Day_of_Week", "Is_Weekend", "Has_Promotion", "Price"]
	target = "Daily_Demand"

	X = data[features]
	y = data[target]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = RandomForestRegressor(random_state=42)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	mae = mean_absolute_error(y_test, predictions)
	print(f"MAE: {mae:.4f}")

	joblib.dump(model, models_dir / "demand_model.joblib")


if __name__ == "__main__":
	main()
