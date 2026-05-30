from pathlib import Path

import pickle

import numpy as np

from inventory_env import InventoryEnv


def get_discrete_state(state):
	return (int(round(state[0])), int(round(state[1])))


def main() -> None:
	env = InventoryEnv()
	base_dir = Path(__file__).resolve().parent
	models_dir = base_dir / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	episodes = 2000
	alpha = 0.1
	gamma = 0.95
	epsilon = 1.0
	epsilon_decay = 0.995
	epsilon_min = 0.01

	q_table = {}

	for episode in range(1, episodes + 1):
		state, _ = env.reset()
		discrete_state = get_discrete_state(state)
		total_reward = 0.0
		terminated = False

		while not terminated:
			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				if discrete_state not in q_table:
					q_table[discrete_state] = np.zeros(env.action_space.n)
				action = int(np.argmax(q_table[discrete_state]))

			next_state, reward, terminated, truncated, _ = env.step(action)
			next_discrete_state = get_discrete_state(next_state)

			if discrete_state not in q_table:
				q_table[discrete_state] = np.zeros(env.action_space.n)
			if next_discrete_state not in q_table:
				q_table[next_discrete_state] = np.zeros(env.action_space.n)

			best_next_action = np.argmax(q_table[next_discrete_state])
			td_target = reward + gamma * q_table[next_discrete_state][best_next_action]
			td_error = td_target - q_table[discrete_state][action]
			q_table[discrete_state][action] += alpha * td_error

			discrete_state = next_discrete_state
			total_reward += reward

			if truncated:
				break

		if epsilon > epsilon_min:
			epsilon = max(epsilon_min, epsilon * epsilon_decay)

		if episode % 200 == 0:
			print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

	with open(models_dir / "q_table.pkl", "wb") as f:
		pickle.dump(q_table, f)


if __name__ == "__main__":
	main()
