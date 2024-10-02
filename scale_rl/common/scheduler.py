import numpy as np

EPS = 1e-8


def cyclic_exponential_decay_scheduler(
    decay_period, initial_value, final_value, reverse=False
):
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value + EPS)
    end = np.log(final_value + EPS)

    def scheduler(step):
        cycle_length = decay_period
        cycle_step = step % cycle_length

        steps_left = decay_period - cycle_step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value) - EPS
        if reverse:
            new_value = 1 - new_value
        return new_value

    return scheduler


def linear_decay_scheduler(decay_period, initial_value, final_value):
    def scheduler(step):
        # Ensure step does not exceed decay_period
        step = min(step, decay_period)

        # Calculate the linear interpolation factor
        fraction = step / decay_period
        new_value = (1 - fraction) * initial_value + fraction * final_value

        return new_value

    return scheduler


def constant_value_scheduler(value):
    """
    Returns a scheduler function that always returns the same value.

    Args:
        value (float): The constant value to return.

    Returns:
        function: A scheduler function that always returns `value`.
    """

    def scheduler(step):
        return value

    return scheduler
