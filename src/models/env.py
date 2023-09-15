class VehicleRouter():
   def __init__(n_agents, n_points):
      # For each point: x,y , mag,dir (in relation to starting point 0)
      self.map = T.zeros([n_points, 4])
      # For each agent: Source, Destination
      self.journey = T.zeros([n_agents, 2])

   def step(action):
      #TODO
      return new_state, reward

   def calc_reward(state, new_state):
      new_state
