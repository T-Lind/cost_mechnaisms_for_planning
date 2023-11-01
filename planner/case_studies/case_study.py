from abc import abstractmethod


class CaseStudy:
    @abstractmethod
    def simulate(self, show_results=True):
        pass
