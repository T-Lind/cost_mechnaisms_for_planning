# Exploring Alternative Cost Mechanisms for Probabilistic Planning

### Author: T-Lind

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

## Credits

Hazhar Rahmani for the initial "Planning to Chronicle" paper and codebase.
