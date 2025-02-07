API
==================================

Initialization
---------------------
Initial states of a simulation consist of particle positions and velocities (by default, zero).
You will need to feed these values to `tensorflow` as inputs, e.g.:

.. code-block:: python

    initial_velocity = np.zeros(shape=[1, num_particles, 2])
    initial_position = # Initial positions of your particles

    feed_dict = {
        sim.initial_state.velocity:
            initial_velocity,
        sim.initial_state.position:
            initial_positions,
        sim.initial_state.deformation_gradient:
            identity_matrix +
            np.zeros(shape=(sim.batch_size, num_particles, 1, 1)),
    }
    loss_evaluated, _ = sess.run([loss, opt], feed_dict=feed_dict)


Defining a controller based on simulation states
---------------------------------------------------------

A simulation consists of a series of states, one per time step.

You can get the (symbolic) simulation from `sim.states` to define the loss.

.. code-block:: python

    # This piece of code is for illustration only. I've never tested it.
    # See jump.py for a working example.

    from simulation import Simulation

    # The controller takes previous states as input and generates an actuation
    # and debug information
    def controller(previous_state):
        average_y = tf.reduce_sum(previous_state.position[:, :, 1], axis=(1,), keepdims=False)
        # A stupid controller that actuates according to the average y-coord of particles
        act = 5 * average_y - 1
        debug = {'actuation': act}
        zeros = tf.zeros(shape=(1, num_particles))
        act = act[None, None]
        # kirchhoff_stress stress
        act = E * make_matrix2d(zeros, zeros, zeros, act)
        # Convert to Kirchhoff stress
        actuation = matmatmul(
            act, transpose(previous_state['deformation_gradient']))
        return actuation, debug

    sim = Simulation(
        num_particles=num_particles,
        num_time_steps=12,
        grid_res=(25, 25),
        controller=controller,
        gravity=(0, -9.8),
        E=E) # Particle Young's modulus

    # Fetch, i.e. the final state
    state = sim.states[-1]
    # Particle Positions
    # Array of float32[batch, particle, dimension=0,1]
    state['position']

    # Particle Velocity
    # Array of float32[batch, particle, dimension=0,1]
    state['velocity'] # Array of float32[batch, particle, dimension=0,1]

    # Particle Deformation Gradients
    # Array of float32[batch, particle, matrix dimension0=0,1, matrix dimension1=0,1]
    state['deformation_gradient']
