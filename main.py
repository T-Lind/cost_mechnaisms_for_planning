import json
from timeit import default_timer as timer

import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

from ptcr2.fom import FOM

config_file = 'samples/wedding_fom.json'

with open(config_file) as f:
    spec = json.loads(f.read())

wedding_fom_cost = FOM()
start = timer()
wedding_fom_cost.compute_optimal_policy(spec, cost_based=True)
end = timer()

print('Time elapsed to compute cost-based optimal policy: ', end - start)

wedding_fom_no_cost = FOM()
start = timer()
wedding_fom_no_cost.compute_optimal_policy(spec)
end = timer()

print('Time elapsed to compute no-cost optimal policy: ', end - start)

# Load the two models
wedding_fom_cost = FOM.load('saves/wedding_fom_cost.pkl')
wedding_fom_no_cost = FOM.load('saves/wedding_fom_no_cost.pkl')

# Example run
result_no_cost = wedding_fom_no_cost.simulate_general_and_greedy_algorithms()
print(result_no_cost)
result_cost = wedding_fom_cost.simulate_general_and_greedy_algorithms()
print(result_cost)

# Now, simulate the algorithms across 1000 runs
n_runs = 1_000

no_cost_steps = []
no_cost_costs = []

cost_steps = []
cost_costs = []

for _ in tqdm(range(n_runs)):
    result_no_cost = wedding_fom_no_cost.simulate()
    no_cost_steps.append(result_no_cost['steps'])
    no_cost_costs.append(result_no_cost['total_cost'])

    result_cost = wedding_fom_cost.simulate()
    cost_steps.append(result_cost['steps'])
    cost_costs.append(result_cost['total_cost'])

# Print 5-number summary + mean for general and greedy steps taken
print('Costless Algorithm Steps Summary:')
print('Min:', np.min(no_cost_steps))
print('Q1:', np.percentile(no_cost_steps, 25))
print('Median:', np.median(no_cost_steps))
print('Mean:', np.mean(no_cost_steps))
print('Q3:', np.percentile(no_cost_steps, 75))
print('Max:', np.max(no_cost_steps))

print('\nCost-Optimized Algorithm Steps Summary:')
print('Min:', np.min(cost_steps))
print('Q1:', np.percentile(cost_steps, 25))
print('Median:', np.median(cost_steps))
print('Mean:', np.mean(cost_steps))
print('Q3:', np.percentile(cost_steps, 75))
print('Max:', np.max(cost_steps))

# Print 5-number summary + mean for general and greedy costs incurred
print('\nCostless Algorithm Costs Summary:')
print('Min:', np.min(no_cost_costs))
print('Q1:', np.percentile(no_cost_costs, 25))
print('Median:', np.median(no_cost_costs))
print('Mean:', np.mean(no_cost_costs))
print('Q3:', np.percentile(no_cost_costs, 75))
print('Max:', np.max(no_cost_costs))

print('\nCost-Optimized Algorithm Costs Summary:')
print('Min:', np.min(cost_costs))
print('Q1:', np.percentile(cost_costs, 25))
print('Median:', np.median(cost_costs))
print('Mean:', np.mean(cost_costs))
print('Q3:', np.percentile(cost_costs, 75))
print('Max:', np.max(cost_costs))

alpha = 0.05

# We want to check if both the number of steps and cost is lower for the cost-optimized algorithm than the costless algorithm
t_stat, p_val = ttest_ind(cost_steps, no_cost_steps, equal_var=False, alternative='less')
print("Performing 2-sample t-test to compare costless and cost-optimized algorithms:")
print('T-Statistic:', t_stat)
print('P-Value:', p_val)

if p_val < alpha:
    print(
        'Reject the null hypothesis: The cost-optimized algorithm is significantly lower in steps than the costless algorithm.')
else:
    print(
        'Fail to reject the null hypothesis: The cost-optimized algorithm is not significantly lower in steps than the costless algorithm.')

# Perform an independent 2-sample t-test to determine if the cost-optimized algorithm is significantly lower in cost than the costless algorithm
t_stat, p_val = ttest_ind(cost_costs, no_cost_costs, equal_var=False, alternative='less')
print("\nPerforming 2-sample t-test to compare costless and cost-optimized algorithms:")
print('T-Statistic:', t_stat)
print('P-Value:', p_val)

if p_val < alpha:
    print(
        'Reject the null hypothesis: The cost-optimized algorithm is significantly lower in cost than the costless algorithm.')
else:
    print(
        'Fail to reject the null hypothesis: The cost-optimized algorithm is not significantly lower in cost than the costless algorithm.')
