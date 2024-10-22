import numpy as np
import random
import math
from scipy.stats import norm
import Distribution


# Goal: Find arm with max reward
def CombinatorialSuccessiveEliminationFramework(n_arms, k, B, dists, setting, algo, best_arm, borda_winner,
                                                second_best_mean, eps, borda_winner_scale, dependence_on_q, lower_bound,
                                                upper_bound, lower_scale_bound, upper_scale_bound, experiment, rts):
    # initialize internal statistics
    active_arms = np.arange(n_arms)
    r = 0
    n_pulls = 0
    if experiment == "AC": # measure overall runtime of AC experiment
        runtime = 0
    while len(active_arms) >= k:
        # make random partitions of size k
        random.shuffle(active_arms)
        partitions = []
        rest = []
        for j in range(int(len(active_arms) / k)):
            partitions.append(active_arms[k * j: k * (j + 1)])
        for j in range(len(active_arms) % k):
            rest.append(active_arms[-(j + 1)])
        # pull arms
        active_arms = []
        for partition in partitions:
            if dependence_on_q:
                dists, second_best_mean = Distribution.make_random_normal_dists_for_q(partition, dists, best_arm, eps, lower_bound, upper_bound, lower_scale_bound, upper_scale_bound)
            # initialize internal statistics for query set
            wins = np.zeros(len(partition))
            if setting == "median" or setting == "std":
                wins = [[] for a in partition]
            pulls = np.zeros(len(partition))
            if algo == "CRSH":
                l = 1
                remaining_arms = partition
                # additional while loop for CRSH
                while len(remaining_arms) > 1:
                    # compute b_r
                    b = compute_br(B, n_arms, k, r, l, algo)
                    # at least one pull per query set
                    if b == 0:
                        b = 1
                    # pull query set b times
                    for pull in range(b):
                        if experiment == "AC":
                            s = pull_query_AC(n_pulls, partition, rts)
                            runtime += np.max(s)
                        else:
                            s = pull_query(partition, dists, borda_winner, best_arm, second_best_mean, eps, borda_winner_scale)
                        # update statistics
                        statistics = update_statistics(partition, s, setting, wins, pulls)
                        n_pulls += 1
                    # discard worse arms
                    remaining_arms = arm_elimination(remaining_arms, statistics, partition, algo, 0)
                    l += 1
                active_arms.append(remaining_arms)
            else:
                # compute b_r
                b = compute_br(B, n_arms, k, r, 0, algo)
                # at least one pull per query set
                if b == 0:
                    b = 1
                # pull query set b times
                for pull in range(b):
                    if experiment == "AC":
                        s = pull_query_AC(n_pulls, partition, rts)
                        runtime += np.max(s)
                    else:
                        s = pull_query(partition, dists, borda_winner, best_arm, second_best_mean, eps,
                                       borda_winner_scale)
                    # update statistics
                    statistics = update_statistics(partition, s, setting, wins, pulls)
                    n_pulls += 1
                # discard worse arms
                active_arms = arm_elimination(active_arms, statistics, partition, algo, 0)
        # keep rest arms and tidy up array
        active_arms.append(rest)
        active_arms = np.concatenate(active_arms).ravel()
        active_arms = np.array(active_arms, dtype=int)
        # stop if Budget is reached
        if n_pulls >= B:
            if experiment == "AC":
                return random.choice(active_arms), runtime
            return random.choice(active_arms)
        r += 1
    # bring rest arms to size k
    if len(active_arms) < k:
        while len(active_arms) < k:
            z = random.choice(np.arange(n_arms))
            if z not in active_arms:
                active_arms = np.append(active_arms, z)
    # reduce from k to 1 active arm
    while len(active_arms) > 1:
        # initialize internal statistics
        wins = np.zeros(len(active_arms))
        if setting == "median" or setting == "std":
            wins = [[] for a in active_arms]
        pulls = np.zeros(len(active_arms))
        if algo == "CRSH":
            l = 1
            remaining_arms = active_arms
            while len(remaining_arms) > 1:
                # compute b_r
                b = compute_br(B, n_arms, k, r, l, algo)
                # at least one pull per query set
                if b == 0:
                    b = 1
                # pull query set b times
                for pull in range(b):
                    if experiment == "AC":
                        s = pull_query_AC(n_pulls, active_arms, rts)
                        runtime += np.max(s)
                    else:
                        s = pull_query(active_arms, dists, borda_winner, best_arm, second_best_mean, eps,
                                       borda_winner_scale)
                    # update statistics
                    statistics = update_statistics(active_arms, s, setting, wins, pulls)
                    n_pulls += 1
                    # stop if Budget is reaches
                    if n_pulls >= B:
                        if experiment == "AC":
                            return random.choice(active_arms), runtime
                        return random.choice(active_arms)
                # discard worst arms
                remaining_arms = arm_elimination(remaining_arms, statistics, active_arms, algo, 0)
                l += 1
            active_arms = [remaining_arms]
        else:
            # compute b_r times
            b = compute_br(B, n_arms, k, r, 0, algo)
            # at least one pull per query set
            if b == 0:
                b = 1
            # pull query set b times
            for pull in range(b):
                if experiment == "AC":
                    s = pull_query_AC(n_pulls, active_arms, rts)
                    runtime += np.max(s)
                else:
                    s = pull_query(active_arms, dists, borda_winner, best_arm, second_best_mean, eps,
                                   borda_winner_scale)
                # update statistics
                statistics = update_statistics(active_arms, s, setting, wins, pulls)
                n_pulls += 1
                # stop if Budget is reached
                if n_pulls >= B:
                    if experiment == "AC":
                        return random.choice(active_arms), runtime
                    return random.choice(active_arms)
            # discard worse arms
            active_arms = arm_elimination(active_arms, statistics, active_arms, algo, 1)
        r += 1
    # return best arm
    if experiment == "AC":
        return active_arms[0], runtime
    return active_arms[0]


def compute_br(B, n_arms, k, r, l, algo):
    # compute individual budget per round and query set
    if algo == "CSWS":
        return int(B / (math.ceil(n_arms / (k ** r)) * (math.ceil(np.log(n_arms) / np.log(k)) + 1)))
    if algo == "CSR":
        return int(B / (math.ceil((n_arms * ((1 - 1 / k) ** (r-1))) / k) * (math.ceil(np.log(1 / n_arms) / np.log((k - 1) / k)) + k - 1)))
    if algo == "CSH":
        return int(B / (math.ceil(n_arms / (k * (2 ** (r-1)))) * (
                    math.ceil(np.log(n_arms) / np.log(2)) + math.ceil(np.log(k) / np.log(2)))))
    if algo == "CRSH":
        return int(B / ((math.ceil(n_arms / (k ** r)) + 1) * math.ceil(np.log(n_arms) / np.log(k)) * (2 ** (l - 1)) * (
                    2 ** math.ceil(np.log(k) / np.log(2)) - 1)))


def arm_elimination(actives, stats, partition, algo, loop):
    # CSWS: keep best arm
    if algo == "CSWS":
        if loop == 0:
            actives.append([partition[np.argmax(stats)]])
        else:
            actives = [partition[np.argmax(stats)]]
    # CSR: reject worst arm
    elif algo == "CSR":
        best_arms = np.delete(partition, np.argmin(stats))
        if loop == 0:
            actives.append(best_arms)
        else:
            actives = best_arms
    # CSH: discard worse half
    elif algo == "CSH":
        median = np.median(stats)
        keep = partition[np.where(stats >= median)]
        while not (int(len(keep)) == int(math.ceil(len(partition)/2)) or int(len(keep)) == int(math.floor(len(partition)/2))):
            arm_to_discard = random.choice(partition[np.where(stats == np.min(stats[np.array([np.where(partition == i)[0] for i in keep], dtype=int).ravel()]))])
            while arm_to_discard not in keep:
                arm_to_discard = random.choice(partition[np.where(stats == np.min(stats[np.array([np.where(partition == i)[0] for i in keep], dtype=int).ravel()]))])
            keep = np.delete(keep, np.where(keep == arm_to_discard))
        if loop == 0:
            actives.append(keep)
        else:
            actives = keep
    # CRSH: discard worse half from remaining arms
    elif algo == "CRSH":
        stats_for_remaining_arms = []
        for arm in actives:
            stats_for_remaining_arms.append(stats[list(partition).index(arm)])
        median = np.median(stats_for_remaining_arms)
        actives = np.delete(actives, np.where(np.array(stats_for_remaining_arms) < median))
    else:
        raise Exception("Algorithm is not defined! Please choose one from {CSWS, CSR, CSH, CRSH}")
    # return still active arms
    return actives


def update_statistics(partition, s, setting, wins, pulls):
    # reward setting
    if setting == "reward":
        stats = np.zeros(len(partition))
        for arm in range(len(partition)):
            pulls[arm] += 1
            wins[arm] += s[arm]
            stats[arm] = wins[arm] / pulls[arm]
    # preference-based setting
    if setting == "pb":
        stats = np.zeros(len(partition))
        for arm in range(len(partition)):
            pulls[arm] += 1
            if s[arm] == np.max(s):
                wins[arm] += 1
            else:
                wins[arm] += 0
            stats[arm] = wins[arm] / pulls[arm]
    # median
    if setting == "median":
        stats = np.zeros(len(partition))
        for arm in range(len(partition)):
            pulls[arm] += 1
            wins[arm].append(s[arm])
            stats[arm] = np.median(wins[arm])
    # std
    if setting == "std":
        stats = np.zeros(len(partition))
        for arm in range(len(partition)):
            pulls[arm] += 1
            wins[arm].append(s[arm])
            stats[arm] = np.std(wins[arm])
    # power mean
    if setting == "power-mean":
        q = 2
        stats = np.zeros(len(partition))
        for arm in range(len(partition)):
            pulls[arm] += 1
            wins[arm] += s[arm]**q
            stats[arm] = (wins[arm] / pulls[arm])**(1/q)
    return stats


def pull_query(partition, dists, borda_winner, condorcet_winner, second_best_mean, eps, scale):
    # sample for each arm from partition a reward from dists
    rewards = []
    for arm in partition:
        if arm == borda_winner and condorcet_winner not in partition:
            rewards.append(norm.rvs(loc=second_best_mean + 2*eps, scale=scale).flatten())
        else:
            rewards.append(dists[arm].sample())
    return rewards

def pull_query_AC(pull, partition, running_times):
    # sample for each arm from partition a reward from dists
    return running_times[pull, partition]
