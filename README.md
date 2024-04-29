# Probabilistic Planning for Optimal Policies

### Author: T-Lind

New to the project? Skip to the [Overview](#introduction) section for a brief introduction.

## Local Installation
Prerequisites:
- Python 3.9 (*later versions do not work*)
- pip

To install the required packages, run the following command:
```bash
# Create venv
python -m venv venv
# Activate venv
source venv/bin/activate
# Install requirements
pip install -r requirements.txt
```

## Docker Installation
Prerequisites:
- Docker

To build the Docker image using the source code, run the following command:
```bash
docker build -t <image_name> .
```

To run the Docker container, run the following command:
```bash
docker run -it <image_name>
```


## Introduction
This project, developed by T-Lind, is an implementation and extension of the work by Dylan A. Shell and Hazhar Rahmani on optimal policies for narrative observation of unpredictable events. It focuses on scenarios where a robot, or any observer, has to monitor an uncertain process over time, choosing what to pay attention to based on a set of constraints.

The project introduces a cost-based optimization approach, where the optimal policy is the one that reduces the most amount of cost while still satisfying the constraints of the problem. This could mean minimizing the distance a robot travels, reducing its wear and improving its efficiency, or minimizing the amount of time it takes to complete a certain task.

The cost is implemented in a matrix, so that the transition from state to state in the model incurs a certain cost. This is then paired with an optimization algorithm that factors in the state-state optimization, allowing the robot to minimize the cost it incurs while still satisfying the constraints of the problem.

This project is useful for developers and researchers working on robotics, automation, and any field where decision-making under uncertainty is crucial. It provides a practical approach to probabilistic planning, with a focus on cost optimization.

## Overview
This work implements and extends the work of Dylan A. Shell and Hazhar Rahmani in their paper ["Planning to chronicle: Optimal policies for narrative observation of unpredictable events"](https://journals.sagepub.com/doi/pdf/10.1177/02783649211069154).
An excerpt from the paper is as follows:

"An important class of applications entails a robot monitoring, scrutinizing, or recording the evolution of an uncertain
time-extended process. This sort of situation leads to an interesting family
of planning problems in which the robot is limited in what it sees and must, thus, choose what to pay attention to.
The distinguishing characteristic of this setting is that the robot has influence over what it  captures via its sensors,
but exercises no causal authority over the evolving process. As such, the robot's objective is to observe the underlying
process and to produce a 'chronicle' of current events, subject to a
goal specification of the sorts of event sequences that may be of interest.
This paper examines variants of such problems when the robot aims to
collect sets of observations to meet a rich specification of their sequential structure. We study this class of problems by modeling a stochastic
process via a variant of a hidden Markov model, and specify the event
sequences of interest as a regular language, developing a vocabulary of 'mutators'
that enable sophisticated requirements to be expressed. Under different suppositions about the information gleaned about the event
model, we formulate and solve different planning problems. The core underlying idea is the construction of a product between the event model
and a specification automaton.

"A concrete motivation for this sort of setting, consider the
proliferation of home videos. These videos are, with remarkably few exceptions,
crummy specimens of the cinematic arts. They fail, generally, to establish and
then bracket a scene; they often founder in emphasizing the importance of key
subjects within the developing action, and are usually unsuccessful in attempts
to trace an evolving narrative arc. And the current generation of autonomous
personal robots and video drones, in their roles as costly and glorified 'selfie sticks'
are set to follow suit. The trouble is that capturing footage to tell a story
is challenging. A camera can only record what you point it toward, so part of
the difficulty stems from the fact that you can't know exactly how the scene will
unfold before it actually does. Moreover, what constitutes structure isn't easily
summed up with a few trite quantities. Another part of the challenge, of course,
is that one has only limited time to capture video footage.
Setting aside pure vanity as a motivator, many applications can be cast as
the problem of producing a finite-length sensor-based recording of the evolution
of some process. As the video example emphasizes, one might be interested
in recordings that meet rich specifications of the event sequences that are of
interest. When the evolution of the event-generating process is uncertain/nondeterministic and sensing is local (necessitating its active direction), then one
encounters an instance from this class of problem. The broad class encompasses
many monitoring and surveillance scenarios. An important characteristic of such
settings is that the robot has influence over what it captures via its sensors, but
cannot control the process of interest.
Our incursion into this class of problem involves two lines of attack. The first
is a wide-embracing formulation in which we pose a general stochastic model,
including aspects of hidden/latent state, simultaneity of event occurrence, and
various assumptions on the form of observability. Secondly, we specify the sequences of interest via a deterministic finite automaton (DFA), and we define
several language mutators, which permit composition and refinement of specification DFAs, allowing for rich descriptions of desirable event sequences. The two
parts are brought together via our approach to planning: we show how to compute an optimal policy (to satisfy the specifications as quickly as possible) via
a form of product automaton. Empirical evidence from simulation experiments
attests to the feasibility of this approach.
Beyond the pragmatics of planning, a theoretical contribution of the paper
is to prove a result on representation independence of the specifications. That
is, though multiple distinct DFAs may express the same regular language and
despite the DFA being involved directly in constructing the product automaton
used to solve the planning problem, we show that it is merely the language expressed that affects the resulting optimal solution. Returning to mutators that
transform DFAs, enabling easy expression of sophisticated requirements, we distinguish when mutators preserve representational independence too."

Additionally, a cost-based optimization approach has been implemented, such that the optimal policy becomes the one that
reduces the most amount of cost, while still satisfying the constraints of the problem. This could mean minimizing the
distance a robot travels, reducing its wear and improving its efficiency, minimizing the amount of time it takes to
complete a certain task.

This "cost" is implemented in a matrix, so that the transition from state to state in the model incurs a certain cost.
This is then paired with an optimization algorithm that factors in the state-state optimization, such that the robot
can minimize the cost it incurs while still satisfying the constraints of the problem. This could be described as a
"navigation" cost.



## Example Usage
In the example usage below, the state names must be specified as strings inside of a list, like seen below.
The transition matrix then must be square of the size of the state names; the row is "from" and the column is "to".
This represents probabilities in the model, so each row should sum up to 1.0. The cost matrix is also square, and
represents the cost of transitioning from one state to another as described above. Note that the costs could be from any
range, but should be consistent across the matrix. The initial distribution is the probability of starting in each state,
and should sum up to 1.0. The state events are the events that can occur in each state, and should be specified as a list
of lists of strings. The single initial states are the states that the robot can start in, and should be specified as a
list of lists of lists, where the innermost list is the state and the outermost list is the initial state. The alphabet is
the list of all possible events that can occur in the model.
```json
{
  "state_names": ["I", "E", "B", "C", "D", "S"],
  "transition_matrix": [
    [0.0, 0.1, 0.3, 0.1, 0.2, 0.3],
    [0.0, 0.1, 0.2, 0.1, 0.3, 0.3],
    [0.0, 0.2, 0.1, 0.1, 0.3, 0.3],
    [0.0, 0.1, 0.2, 0.2, 0.3, 0.2],
    [0.0, 0.2, 0.3, 0.1, 0.1, 0.3],
    [0.0, 0.4, 0.2, 0.2, 0.2, 0.0]
  ],
  "cost_matrix": [
    [1, 5, 2, 3, 2, 1],
    [1, 1, 5, 2, 3, 4],
    [2, 1, 1, 1, 2, 3],
    [3, 2, 5, 1, 1, 4],
    [2, 5, 2, 5, 1, 4],
    [5, 2, 3, 2, 4, 1]
  ],

  "initial_distribution": [0.0, 0.1, 0.3, 0.2, 0.2, 0.2],
  "state_events": [
      [[], ["e1"], ["b1"], ["c1"], ["d1"], ["s1"]],
      [[], ["e2"], ["b2"], ["c2"], ["d2"], ["s2"]],
      [[], ["e3"], ["b3"], ["c3"], ["d3"], ["s3"]]
    ],
  "single_initial_states": [
    [[["d1", "d2"], "d12"]],
    [[["d2", "d3"], "d23"]]
  ],

  "alphabet": ["e1", "b1", "c1", "d1", "s1", "e2", "b2", "c2", "d2", "s2", "d12", "e3", "b3", "c3", "d3", "s3", "d23"],
```

Demonstrate capabilities:
```python
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
```

## Questions?

If you have any questions, feel free to reach out to me at [tiernanlind@tamu.edu](mailto:tiernanlind@tamu.edu).

## Acknowledgements

Dylan A. Shell


Hazhar Rahmani
