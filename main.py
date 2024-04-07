import json
import time
from timeit import default_timer as timer

from ptcr2.FOM import FOM

config_file = 'samples/wedding_fom.json'
# parse file name
file_name = config_file.split('/')[-1].split('.')[0]

if __name__ == "__main__":
    with open(config_file) as f:
        spec = json.loads(f.read())

    story_counts = {}

    n_simulations = 10_000

    n_steps_list = []
    recorded_times = []
    expected_list = []

    wedding_fom = FOM()
    for i in range(n_simulations):
        start = timer()

        results = wedding_fom.simulate(spec)
        run_number_of_steps = results['run_number_of_steps']
        expected_number_of_steps = results['expected']
        recorded_story = results['recorded_story']
        diff_tracker = results['diff_tracker']

        if recorded_story in story_counts:
            story_counts[recorded_story] += 1
        else:
            story_counts[recorded_story] = 1

        end = timer()
        recorded_times.append(end - start)
        n_steps_list.append(run_number_of_steps)
        expected_list.append(expected_number_of_steps)

    print("Number of iterations to compute optimal policy", len(diff_tracker))
    print("Expected average:", sum(expected_list) / n_simulations)
    print("Average number of steps:", sum(n_steps_list) / n_simulations)
    print("Average time:", sum(recorded_times) / n_simulations)

    if spec['log']:
        # save data to csv file
        import csv

        with open(file_name + '_' + str(int(time.time())) + '.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['n_steps_real', 'n_steps_expected', 'time'])
            rows = list(zip(n_steps_list, expected_list, recorded_times))
            writer.writerows(rows)

    if spec['plot']:
        import matplotlib.pyplot as plt

        # create four histograms in one window
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(diff_tracker, label='Calculated Difference')
        plt.legend()
        plt.ylabel("Difference")
        plt.xlabel("Iteration")
        plt.title('Optimal Policy Calculation')
        plt.subplot(2, 2, 2)
        plt.hist(n_steps_list, bins=20)
        plt.title(f'Actual Number of Steps ({n_simulations} simulations)')
        plt.ylabel("Frequency")
        plt.xlabel("Number of Steps")
        plt.subplot(2, 2, 3)
        plt.hist(expected_list, bins=20, range=(30.11, 30.13))
        plt.title(f'Expected Number of Steps ({n_simulations} simulations)')
        plt.ylabel("Frequency")
        plt.xlabel("Number of Steps")
        plt.subplot(2, 2, 4)
        plt.hist(recorded_times, bins=20, range=(0.0005, 0.0015))
        plt.title(f'Compute Time ({n_simulations} simulations)')
        plt.ylabel("Frequency")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
