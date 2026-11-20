import math

import torch


def carrier_phase(frequency, travel_time):
    return 2.0 * math.pi * frequency * travel_time


def phase_factors(phase):
    return torch.cos(phase), torch.sin(phase)


def rotate_components(real, imag, cosine, sine):
    return real * cosine - imag * sine, real * sine + imag * cosine


def phase_factors_from_travel_time(frequency, travel_time):
    return phase_factors(carrier_phase(frequency, travel_time))


def receive_phase_factors(frequency, total_travel_time, transmit_travel_time):
    return phase_factors_from_travel_time(
        frequency, total_travel_time - transmit_travel_time
    )


def rotate_iq_by_travel_time(real, imag, frequency, travel_time):
    cosine, sine = phase_factors_from_travel_time(frequency, travel_time)
    return rotate_components(real, imag, cosine, sine)
