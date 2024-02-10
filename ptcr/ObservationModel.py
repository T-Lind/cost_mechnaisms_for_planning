from typing import Set, Dict
import json
import numpy as np

class ObservationModel:
    def __init__(self, observations: Set[str], emission_probabilities: Dict[str, float]):
        self.observations = observations
        self.emission_probabilities = emission_probabilities

    def emit_observation(self, current_state: str):
        """
        Emit an observation based on the current state.
        :param current_state: The current state of the event model.
        :return: The emitted observation.
        """
        observation_probs = self.emission_probabilities.get(current_state, {})
        if not observation_probs:
            return None

        observation = np.random.choice(list(observation_probs.keys()), p=list(observation_probs.values()))
        return observation

    @classmethod
    def load_model(cls, observation_model_raw: str):
        # Loads the observation model from a JSON string
        observation_model_data = json.loads(observation_model_raw)

        observations = observation_model_data["observations"]
        emission_probabilities = observation_model_data["emission_probabilities"]

        return cls(observations, emission_probabilities)
