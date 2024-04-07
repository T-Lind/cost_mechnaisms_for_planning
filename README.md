# Exploring Alternative Cost Mechanisms for Probabilistic Planning

### Author: T-Lind

## Installation
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

## Overview

The "Planning to Chronicle" paper proposed an approach to probabilistic planning that minimizes the expected number of
steps required to reach a goal state (Rahmani et al.). However, in different scenarios, the criteria for optimality
might differ from simply reducing the expected number of steps. This research aims to explore alternative cost
mechanisms that could improve performance for certain problems.

Rather than counting steps, actions will be associated with cost values that capture domain-specific preferences. The
planner will minimize cumulative cost to find lower-cost plans. Defining appropriate costs will allow encoding soft
constraints and tradeoffs not representable by step counts alone.

To enable this research, the "Planning to Chronicle" codebase will be refactored with improved architecture and
documentation. Cost mechanisms and related algorithms will then be implemented and evaluated on testing scenarios.
Comparisons to the step-minimizing approach will determine when cost-based planning produces better solutions.

This research will provide new insights into designing cost functions that capture planning objectives beyond step
minimization. More flexible cost definitions could expand the range of problems addressed by probabilistic planning
techniques. The refactored code will also improve maintainability and extensibility for future work.

## Smart Home Ideas

- Event Model:
    - Random events such as temperature changes, motion detection, or appliance malfunctions in the house.
    - Examples: Sudden increase in temperature, a motion detected in a room, a door left open, or a malfunction in a
      smart device. Markov Decision Process (MDP):
- Markov Decision Process (MDP):
    - The smart home system's decision-making process for managing devices and responding to events.
    - Actions may include adjusting thermostat settings, turning on/off lights, locking doors, or notifying the
      homeowner about an event.
    - Probabilistic outcomes based on the current state and chosen actions (e.g., adjusting the thermostat may or may
      not result in the desired temperature).
- Deterministic Finite Automaton (DFA):
    - A predefined sequence of tasks representing a daily routine or a specific scenario that the smart home should
      handle.
    - Examples: "Nighttime Security Check," "Guest Arrival," or "Morning Routine." The DFA ensures that certain tasks
      are completed in a specific order for the smart home to function effectively.

## Example
Positional States:
- S_0: Front Yard
- S_1: Back Yard
- S_2: Side of the house
- S_4: Near the front door
- S_5: In the living room
- S_6: In the kitchen
- S_7: In the bedroom
- S_8: In the dining room
- S_9: Near the back door

Event States:
- S_0: Front Door
  - Events: Open, Close
- S_1: Back Door
  - Events: Open, Close
- S_2: Side Door
  - Events: Open, Close
- S_3: Security System Console
    - Events: Malfunction, Normal
- S_4: Security Cameras
    - Events: Activate, Deactivate
- S_5: Temperature Console
    - Events: High, Low
- S_6: Nothing
    - Events: None


DFA States:
- S_0: The smart home is in a normal state. All lights are off, doors and windows closed and locked, and security cameras are active
- S_1: Motion detected in the backyard
- S_2: Motion detected in the front yard
- S_3: Motion detected in the side yard
- S_4: The temperature is too high
- S_5: The temperature is too low
- S_6: The front door is open
- S_7: The back door is open
- S_8: The side door is open
- S_9: The security system has malfunctioned
- S_10: Security check completed

