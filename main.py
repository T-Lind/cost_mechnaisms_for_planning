from ptcr2.FOM import WeddingFOM
from timeit import default_timer as timer

story_counts = {}

n_simulations = 100

n_steps_list = []
recorded_times = []
expected_list = []

for i in range(n_simulations):
    start = timer()

    wedding_fom = WeddingFOM()
    expected, run_number_of_steps, recorded_story, policy_comp_time = wedding_fom.simulate()
    if recorded_story in story_counts:
        story_counts[recorded_story] += 1
    else:
        story_counts[recorded_story] = 1

    end = timer()
    recorded_times.append(end - start)
    n_steps_list.append(run_number_of_steps)
    expected_list.append(expected)

print("Expected average:", sum(expected_list) / n_simulations)
print("Average number of steps:", sum(n_steps_list) / n_simulations)
print("Average time:", sum(recorded_times) / n_simulations)



