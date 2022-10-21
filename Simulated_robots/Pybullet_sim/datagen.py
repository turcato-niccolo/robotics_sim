import numpy as np
from scipy import signal


def get_filtered_wgn_trj(num_samples, dt, num_joints, rng, mean_vec, std_vec, fc):
    """
        Return a joint position and velocity trj obtained by filtering white gaussian noise (wgn)
        with a second order butterworth filter.

        Parameters
        ----------
        num_samples : int
            Number of samples of the trj.
        dt : float
            Sampling period in seconds.
        num_joints : int
            Number of non-fixed joints.
        rng : numpy.random.Generator
            Random number generator obj.
        mean_vec : array of `num_joints` elems
            Mean of the wgn vector.
        std_vect : array of `num_joints` elems
            Standard deviations of the wgn.
        fc : float
            Cut-off frequency of the filter in Hertz.

        Returns
        -------
        q_ref : ndarray of dimentions `num_samples` x `num_joints`
            Cols are trjs of single joints.
        dq_ref : ndarray of dimentions `num_samples` x `num_joints`
            Cols are ref vels of single joints.

        """

    # Get the second order butterworth filter
    b, a = signal.butter(2, fc / 0.5 * dt)
    # Sample the wgn vectors
    wgn = rng.standard_normal([num_samples, num_joints]) * std_vec + mean_vec
    # Compute the trjs
    q_ref = []
    dq_ref = []
    ddq_ref = []
    for i in range(num_joints):
        pos = signal.lfilter(b, a, signal.lfilter(b, a, signal.lfilter(b, a, wgn[:, i])))
        # pos = wgn[:,i]
        # pos=signal.lfilter(b,a,signal.lfilter(b,a,wgn[:,i]))
        # vel = signal.lfilter(b,a,(pos[2:,] - pos[:-2,])/(2*dt))
        vel = (pos[2:, ] - pos[:-2, ]) / (2 * dt)
        acc = (vel[2:, ] - vel[:-2, ]) / (2 * dt)
        pos = pos[1:-1]
        q_ref.append(pos.reshape([-1, 1]))
        dq_ref.append(vel.reshape([-1, 1]))
        ddq_ref.append(acc.reshape([-1, 1]))

    # Return the computed trjs as numpy arrays
    return np.concatenate(q_ref, axis=1), np.concatenate(dq_ref, axis=1),  np.concatenate(ddq_ref, axis=1)


def noising_signals(tau, std_noise, mean_noise):
    """
    Applies random Gaussian Noise to the signal

    :param tau:         Input signal (#samples x dof)
    :param std_noise:   Std dev to use for gaussian noise noise
    :param mean_noise   Means for the Gaussian Noise
    :return: The signal + noise
    """
    noise = np.random.normal(mean_noise, std_noise, tau.shape)

    return tau + noise


def get_sum_of_sinusoids_trj(num_samples, dt, num_joints, rng, max_freq, num_freq, max_q, q_mean, max_dq, max_ddq):
    """
        Return a joint position, velocity and accelertion trj obtained by summing sinusoidal waves of different frequencies.
        For each joint a set of `num_freq` frequencies is sampled from a uniform distribution between `max_freq` and `max_freq`.
        For each joint, the amplitude of each sinusoid is the same and is computed in order not to overcome `max_q` `max_dq` and `max_ddq`

        Parameters
        ----------
        num_samples : int
            Number of samples of the trj.
        dt : float
            Sampling period in seconds.
        num_joints : int
            Number of non-fixed joints.
        rng : numpy.random.Generator
            Random number generator obj.
        max_freq : float
            Maximum frequency.
        num_freq : int
            Number of sinusoids.
        max_q : list
            Maximum amplitude for the for trj of each joint. It should be half the range extension
        q_mean : list
            Mean of the range of each join. Used to translate the position trj.
        max_dq : list
            Maximum amplitude for the velocity trj.
        max_ddq : list
            Maximum amplitude for the acceleration trj.

        Returns
        -------
        q_ref : ndarray of dimentions `num_samples` x `num_joints`
            Cols are trjs of single joints.
        dq_ref : ndarray of dimentions `num_samples` x `num_joints`
            Cols are ref vels of single joints.

        """

    # Construct the time vector
    t = np.linspace(0, dt * (num_samples - 1), num_samples)

    # Sample the frequencies
    freq = []
    for i in range(num_joints):
        freq.append(rng.uniform(low=-max_freq, high=max_freq, size=num_freq))

    # Compute the amplitudes
    weights = []
    for i in range(num_joints):
        a = [max_q[i] / num_freq, max_dq[i] / (2 * np.pi * np.sum(np.abs(freq[i]))),
             max_ddq[i] / (4 * np.pi * np.sum([f ** 2 for f in freq[i]]))]
        weights.append(np.min(a))

    # Compute the sine waves
    q_ref = np.zeros((num_samples, num_joints))
    dq_ref = np.zeros((num_samples, num_joints))
    ddq_ref = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        q_ref[:, i] = weights[i] * np.sum(np.array([np.sin(2 * np.pi * f * t) for f in freq[i]]), axis=0) + q_mean[i]
        dq_ref[:, i] = weights[i] * (2 * np.pi) * np.sum(np.array([f * np.cos(2 * np.pi * f * t) for f in freq[i]]),
                                                         axis=0)
        ddq_ref[:, i] = -weights[i] * ((2 * np.pi) ** 2) * np.sum(
            np.array([(f ** 2) * np.sin(2 * np.pi * f * t) for f in freq[i]]), axis=0)

    # Return the computed trjs
    return q_ref, dq_ref, ddq_ref