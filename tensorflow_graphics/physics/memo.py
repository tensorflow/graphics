class Memo:
  """
  The :class:`Memo` is the data structure that records the result of each
  timestep of simulation and returns it to be interpreted by a higher-level algorithm.
  The key part of the Memo that is of most interest to a user is :attr:`steps`, which
  is a list of Tuples as defined in :mod:`time_integration` in the method :func:`get_name_states`.
  Please note that data from simulation substeps will NOT be returned.
  Other data saved in the Memo includes actuation sequences, simulation initializations, running losses,
  and information for visualization.

  :param steps: A list of tuples representing the computed results of each step of a simulation.  If simulated for n timesteps, steps will be of length n+1, where the first entry is the initial state of the simulation.
  :param actuations: A list of the actuation signal at each step of the simulation.
  :param initial_state: An object of type :class:\`InitialState` which represents the state of the system prior to simulation.
  :param iteration_feed_dict: An optional dictionary to be used for evaluating tensors in the simulation.
  :param point_visualization: An optional list of tuples, each containing a numpy array of points to be visualized followed by color and radius information.  Typically filled in automatically depending on how :meth:`Simulation.add_point_visualization` is called.
  :param vector_visualization: An optional list of tuples, each containing a numpy array of vector directions to be visualized followed by color and and length information.  Typically filled in automatically depending on how :meth:`Simulation.add_vector_visualization` is called.
  """
  #DOCTODO: stepwise loss and return loss if released.

  def __init__(self) -> None:
    """Create a new empty Memo."""
    self.steps = []
    self.actuations = []
    self.initial_state = None
    self.initial_feed_dict = {}
    self.iteration_feed_dict = {}
    self.point_visualization = []
    self.vector_visualization = []
    self.stepwise_loss = None
    self.return_losses = None

  def update_stepwise_loss(self, step):
    #DOCTODO: update if released.
    from IPython import embed
    if step is None:
      return
    if self.stepwise_loss is None:
      self.stepwise_loss = step
      return
    if not isinstance(step, list):
      self.stepwise_loss += step
      return

    def add(a, b):
      if isinstance(a, list):
        for x, y in zip(a, b):
          add(x, y)
      else:
        a += b

    add(self.stepwise_loss, step)
