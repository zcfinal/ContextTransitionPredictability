import math
import numpy as np
import re
import entropy_time_bin

def rand_entropy(sequence):
    """
    Compute the "random entropy", that is, the entropy of a uniform distribution.
    Equation:
        S_{rand} = \log_{2}(n), where n is the number of unique symbols in the input sequence.
    Args:
        sequence: 1-D array-like sequence of symbols.
    Returns:
        A float representing the random entropy of the input sequence.
    Reference:
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu,
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    alphabet_size = len(np.unique(sequence))
    return np.log2(alphabet_size)


def unc_entropy(sequence):
    """
    Compute temporal-uncorrelated entropy (Shannon entropy).
    Equation:
    S_{unc} = - \sum p(i) \log_2{p(i)}, for each symbol i in the input sequence.
    Args:
        sequence: the input sequence of symbols.
    Returns:
        temporal-uncorrelated entropy of the input sequence.
    Reference:
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu,
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    _, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def lambdas_naive(sequence):
    """
    Compute the lambdas in the following equation:

    Equation:
        S_{real} = \left( \frac{1}{n} \sum \Lambda_{i} \right)^{-1}\log_{2}(n)

    Args:
        sequence: the input sequence of symbols.
    Returns:
        The sum of the average length of sub-sequences that
        (up to a certain point) do not appear in the original sequence.
    Reference:
        Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
        Nonparametric entropy estimation for stationary processes and random
        fields, with applications to English text. IEEE Transactions on Information
        Theory, 44(3), 1319-1327.
    """
    lambdas = 0
    for i in range(len(sequence)):
        current_sequence = ','.join(sequence[0:i])
        match = True
        k = i
        while match and k < len(sequence):
            k += 1
            match = ','.join(sequence[i:k]) in current_sequence
        lambdas += (k - i)
    if lambdas==0:
        lambdas=1
    return lambdas


def real_entropy(lambdas, n):
    """
    Estimate the entropy rate of the symbols encoded in the input sequence.

    Equation:
        S_{real} = \left( \frac{1}{n} \sum \Lambda_{i} \right)^{-1}\log_{2}(n)

    Args:
        sequence: the input sequence of symbols.
    Returns:
        A float representing the entropy rate of the input sequence.
    Reference:
        Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
        Nonparametric entropy estimation for stationary processes and random
        fields, with applications to English text. IEEE Transactions on Information
        Theory, 44(3), 1319-1327.
    """
    return (1.0 * n / lambdas) * np.log(n)

def compute_f(p,S,N):
    if p<=0 or p>=1 :
        print(p)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    pi_max = h + (1 - p) * np.log2(N - 1) - S
    return pi_max

def getapproximation(p,S,N) :
    f= compute_f(p,S,N)
    d1 = np.log2(1-p) - np.log2(p) - np.log2(N-1)
    d2 = 1 / ((p-1)*p)
    return f/(d1-f*d2/(2*d1))

def max_predictability(S, N):
    """
    Estimate the maximum predictability of a sequence with
    entropy S and alphabet size N.
    Equation:
    $S = - H(\Pi) + (1 - \Pi)\log(N - 1),$
        where $H(\Pi)$ is given by
    $H(\Pi) = \Pi \log_2(\Pi) + (1 - \Pi) \log_2(1 - \Pi)$
    Args:
        S: the entropy of the input sequence of symbols.
        N: the size of the alphabet (number of unique symbols)
    Returns:
        the maximum predictability of the sequence.
    Reference:
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu,
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    if S>np.log2(N) :
        return 0
    if S<=0.01 :
        return 0.999
    p = (N+1)/(2*N)
    while(abs(compute_f(p,S,N))>0.0000001):
        p = p - 0.8*getapproximation(p,S,N)
    return p


def regularity(sequence):
    """
    Compute the regularity of a sequence.
    The regularity basically measures what percentage of a user's
    visits are to a previously visited place.
    Parameters
    ----------
    sequence : list
        A list of symbols.
    Returns
    -------
    float
        100 minus the percentage of the symbols in the sequence that are unique.
    """
    if len(set(sequence)) <= 1:
        return 100.0

    if len(set(sequence)) == len(sequence):
        return .0

    return round(100.0 - len(set(sequence)) * 100 / len(sequence), 2)


def stationarity(sequence):
    """
    Compute the stationarity of a sequence.
    A stationary transition is one whose source and destination symbols
    are the same. The stationarity measures the percentage of transitions
    to the same location.
    Parameters
    ----------
    sequence : list
        A list of symbols.
    Returns
    -------
    float
        Percentage of the sequence that is stationary.
    """
    if len(sequence) <= 1:
        return 100.0

    if len(sequence) == len(set(sequence)):
        return .0

    stationary_transitions = 0
    for i in range(1, len(sequence)):
        if sequence[i - 1] == sequence[i]:
            stationary_transitions += 1
    return round(stationary_transitions * 100 / (len(sequence) - 1), 2)


def diversity(locations, user_home):
    """
    Compute the of trajectories of a user's mobility trace.
    The diversity of trajectories is the ratio of unique home-home
    trajectories and their sizes compared to the total length of
    the trace.
    Parameters
    ----------
    locations : list
        A list of locations that a user visited.
    user_home : str
        A string representing the ID of the location of the user's home.
    Returns
    -------
    float
        The percentage of the overall trajectories that is accounted for
        by the unique trajectories.
    """
    if not locations:
        return .0

    # We assume that the user's trajectory starts and ends at their home location.
    if not locations[0].startswith(user_home):
        locations = [user_home] + locations
    if not locations[-1].startswith(user_home):
        locations.append(user_home)

    # Split home-to-home trajectories into groups.
    trajectories = re.split('(' + str(user_home) + ')', ''.join(str(loc) for loc in locations))

    # Counts home-to-home trajectories.
    trajectories = [traj for traj in trajectories if traj != '' and traj != user_home]

    if len(set(trajectories)) <= 1:
        return .0

    if len(set(trajectories)) == len(trajectories):
        return 100.0

    # Compute the diversity of trajectories: number of unique trajectories
    # times average size of unique trajectories as a percentage of the
    # total number of trajectories.
    if trajectories:
        unique_trajectories = len(set(trajectories))
        mean_unique_trajectory_size = np.mean([max(1, len(traj) // len(str(user_home))) for traj in set(trajectories)])
        return (unique_trajectories * mean_unique_trajectory_size) * 100 / len(locations)

    return .0


if __name__ == '__main__':
    b= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    a= [str(num) for num in b]
    lam = lambdas_naive(a)
    print(lam)
    print(real_entropy(lam,len(a)),unc_entropy(a))
    print(max_predictability(real_entropy(lam,len(a)),len(np.unique(a))))