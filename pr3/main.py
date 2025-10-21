from pr3.swarm import GlobalSwarm, LocalSwarm
import numpy as np


def drop_wave_function(coords):
    x, y = coords[0], coords[1]
    return (
        -1 * (
            (1 + np.cos(12 * (
                np.sqrt(x ** 2 + y ** 2)
            )))
            /
            (
                0.5 * (x ** 2 + y ** 2) + 2
            )
        )
    )



gswarm = GlobalSwarm(
    function=drop_wave_function,
    fields_from=np.array([-5.12, -5.12]),
    fields_to=np.array([5.12, 5.12]),
    n_particles=300,
    cognitive_coefficient=2,
    social_coefficient=2,
    inertia=0.7,
)

gswarm.optimize(100)

# gswarm = LocalSwarm(
#     function=drop_wave_function,
#     fields_from=np.array([-5.12, -5.12]),
#     fields_to=np.array([5.12, 5.12]),
#     n_particles=300,
#     cognitive_coefficient=2,
#     social_coefficient=2,
#     inertia=0.7,
#     neighbors=1,
# )

# gswarm.optimize(100)