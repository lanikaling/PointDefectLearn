import numpy as np

interstitial_sites = {
    "225": np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.25, 0.25, 0.25],
        ]
    ),
    "194": np.array(
        [
            [0.3333, 0.6666, 0.125],
            [0.3333, 0.6666, 0.25],
            [0.3333, 0.6666, 0.375],
            [0.6667, 0.3334, 0.625],
            [0.6667, 0.3334, 0.75],
            [0.6667, 0.3334, 0.875],
        ]
    ),
    "229": np.array(
        [
            [0.3333, 0.6666, 0.125],
            [0.3333, 0.6666, 0.25],
            [0.3333, 0.6666, 0.375],
            [0.6667, 0.3334, 0.625],
            [0.6667, 0.3334, 0.75],
            [0.6667, 0.3334, 0.875],
        ]
    ),
    "139": np.array(
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.75],
        ]
    ),
    "141": np.array(
        [
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.0],
            [0.25, 0.75, 0.25],
            [0.25, 0.75, 0.5],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.0],
            [0.75, 0.25, 0.25],
            [0.75, 0.25, 0.5],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.0],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.5],
            [0.75, 0.75, 0.75],
            [0.0, 0.25, 0.5],
            [0.0, 0.25, 0.75],
            [0.0, 0.75, 0.5],
            [0.0, 0.75, 0.75],
            [0.25, 0.0, 0.25],
            [0.25, 0.0, 0.5],
            [0.25, 0.5, 0.0],
            [0.25, 0.5, 0.75],
            [0.5, 0.25, 0.0],
            [0.5, 0.25, 0.25],
            [0.5, 0.75, 0.0],
            [0.5, 0.75, 0.25],
            [0.75, 0.0, 0.25],
            [0.75, 0.0, 0.5],
            [0.75, 0.5, 0.0],
            [0.75, 0.5, 0.75],
            [0.125, 0.125, 0.125],
            [0.125, 0.125, 0.875],
            [0.125, 0.375, 0.125],
            [0.125, 0.375, 0.375],
            [0.125, 0.625, 0.125],
            [0.125, 0.625, 0.375],
            [0.125, 0.875, 0.125],
            [0.125, 0.875, 0.875],
            [0.375, 0.125, 0.625],
            [0.375, 0.125, 0.875],
            [0.375, 0.375, 0.375],
            [0.375, 0.375, 0.625],
            [0.375, 0.625, 0.375],
            [0.375, 0.625, 0.625],
            [0.375, 0.875, 0.625],
            [0.375, 0.875, 0.875],
            [0.625, 0.125, 0.625],
            [0.625, 0.125, 0.875],
            [0.625, 0.375, 0.375],
            [0.625, 0.375, 0.625],
            [0.625, 0.625, 0.375],
            [0.625, 0.625, 0.625],
            [0.625, 0.875, 0.625],
            [0.625, 0.875, 0.875],
            [0.875, 0.125, 0.125],
            [0.875, 0.125, 0.875],
            [0.875, 0.375, 0.125],
            [0.875, 0.375, 0.375],
            [0.875, 0.625, 0.125],
            [0.875, 0.625, 0.375],
            [0.875, 0.875, 0.125],
            [0.875, 0.875, 0.875],
            [0.125, 0.125, 0.375],
            [0.125, 0.125, 0.625],
            [0.125, 0.375, 0.625],
            [0.125, 0.375, 0.875],
            [0.125, 0.625, 0.625],
            [0.125, 0.625, 0.875],
            [0.125, 0.875, 0.375],
            [0.125, 0.875, 0.625],
            [0.375, 0.125, 0.125],
            [0.375, 0.125, 0.375],
            [0.375, 0.375, 0.125],
            [0.375, 0.375, 0.875],
            [0.375, 0.625, 0.125],
            [0.375, 0.625, 0.875],
            [0.375, 0.875, 0.125],
            [0.375, 0.875, 0.375],
            [0.625, 0.125, 0.125],
            [0.625, 0.125, 0.375],
            [0.625, 0.375, 0.125],
            [0.625, 0.375, 0.875],
            [0.625, 0.625, 0.125],
            [0.625, 0.625, 0.875],
            [0.625, 0.875, 0.125],
            [0.625, 0.875, 0.375],
            [0.875, 0.125, 0.375],
            [0.875, 0.125, 0.625],
            [0.875, 0.375, 0.625],
            [0.875, 0.375, 0.875],
            [0.875, 0.625, 0.625],
            [0.875, 0.625, 0.875],
            [0.875, 0.875, 0.375],
            [0.875, 0.875, 0.625],
        ]
    ),
    "140": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.75],
            [0.5, 0.5, 0.25],
            [0.5, 0.5, 0.75],
            [0.0, 0.25, 0.25],
            [0.0, 0.25, 0.75],
            [0.0, 0.75, 0.25],
            [0.0, 0.75, 0.75],
            [0.25, 0.0, 0.25],
            [0.25, 0.0, 0.75],
            [0.25, 0.5, 0.25],
            [0.25, 0.5, 0.75],
            [0.5, 0.25, 0.25],
            [0.5, 0.25, 0.75],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.75],
            [0.75, 0.0, 0.25],
            [0.75, 0.0, 0.75],
            [0.75, 0.5, 0.25],
            [0.75, 0.5, 0.75],
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.5],
            [0.25, 0.75, 0.0],
            [0.25, 0.75, 0.5],
            [0.75, 0.25, 0.0],
            [0.75, 0.25, 0.5],
            [0.75, 0.75, 0.0],
            [0.75, 0.75, 0.5],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.75],
            [0.125, 0.375, 0.5],
            [0.125, 0.625, 0.0],
            [0.375, 0.125, 0.0],
            [0.375, 0.875, 0.5],
            [0.625, 0.125, 0.5],
            [0.625, 0.875, 0.0],
            [0.875, 0.375, 0.0],
            [0.875, 0.625, 0.5],
        ]
    ),
}