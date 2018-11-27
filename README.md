# Collision-Avoidance-using-Markov-Decision-Process
Avoiding car collisions using MDP

- State space is a 2D grid with obstacle and destination locations.
- Reward space is in the form of costs (negative rewards) for moving and collision. Destination reward is positive.
- Action space is 4 directions in the 2D grid.

State transition model is probabilistic with higher probability for the intended direction
