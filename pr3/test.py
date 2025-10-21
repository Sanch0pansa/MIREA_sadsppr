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


print(np.round(drop_wave_function(np.array([0.5377, -0.2118])), 4))