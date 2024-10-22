import numpy as np
import math
from CSFramework import pull_query, pull_query_AC


def SuccessiveHalving(n_arms, B, dists, best_arm, borda_winner, second_best_mean, eps, borda_winner_scale, experiment,
                      rts):
    # initialize internal statistics
    active_arms = np.arange(n_arms)
    n_pulls = 0
    pulls = np.zeros(n_arms)
    wins = np.zeros(n_arms)
    if experiment == "AC": # measure overall runtime for AC experiment
        runtime = 0
    for k in range(math.ceil(np.log(n_arms) / np.log(2))):
        stats_for_actives = np.zeros(len(active_arms))
        r_k = math.floor( B / (len(active_arms) * math.ceil(np.log(n_arms) / np.log(2))) )
        for arm_id, arm in enumerate(active_arms):
            # pull each active arm for r_k times
            for pull in range(r_k):
                if experiment == "AC":
                    s = pull_query_AC(n_pulls, [arm], rts)
                    runtime += s[0]
                else:
                    s = pull_query([arm], dists, borda_winner, best_arm, second_best_mean, eps,
                                   borda_winner_scale)
                pulls[arm] += 1
                wins[arm] += s[0]
            # update statistics
            stats_for_actives[arm_id] = wins[arm] / pulls[arm]
        # update active arms
        active_arms = active_arms[np.argsort(stats_for_actives)[int(math.floor(len(stats_for_actives)/2)) : len(stats_for_actives)]]
    if experiment == "AC":
        return active_arms[0], runtime
    return active_arms[0]
