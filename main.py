from planner.case_studies.case_study import CaseStudy
from planner.case_studies.paris_fom import ParisFOM
from timeit import default_timer as timer


class Main:
    studies: dict[str: CaseStudy] = {
        "paris_fom": ParisFOM(),
    }

    @staticmethod
    def run_case_study(case_study_type, results_file_path, n_simulations=1000):
        study_html = """\
        <html>
        <table border='1'>
        <tr>
        <th>Policy Compute Time</th>
        <th>Round</th>
        <th>Expected</th>
        <th>#Steps</th>
        <th>Average</th>
        <th>Recorded</th>
        </tr>
        """

        sum = 0
        sum_time = 0
        story_counts = {}

        sim = Main.studies[case_study_type]

        for i in range(n_simulations):
            study_html += "<tr>"
            start = timer()
            result = sim.simulate()

            expected = result["expected"]
            average = result["average"]
            story = result["story"]
            computation_time = result["computation_time"]

            sum += average
            print("# Of Steps: " + str(average))

            calculated_average = sum / (i + 1)

            if story in story_counts:
                story_counts[story] += 1
            else:
                story_counts[story] = 1

            study_html += "<td>" + str(computation_time) + "</td>"
            study_html += "<td>" + str(i + 1) + "</td>"
            study_html += "<td>" + str(expected) + "</td>"
            study_html += "<td>" + str(average) + "</td>"
            study_html += "<td>" + str(calculated_average) + "</td>"
            study_html += "<td>" + str(story) + "</td>"

            end = timer()
            sum_time += end - start
            study_html += "</tr>"

            calculated_average = sum / n_simulations
            avg_time = sum_time / n_simulations

            print("Expected number of steps is: " + str(expected))
            print("Average number of steps is: " + str(calculated_average))
            print("Average time is: " + str(avg_time))

        study_html += "</table></html>"
        print("Story Counts: " + str(story_counts))

        with open(results_file_path, "w") as file:
            file.write(study_html)


Main.run_case_study("paris_fom", "paris_fom_results.html")
