# Exploring Alternative Cost Mechanisms for Probabilistic Planning

### Author: T-Lind

New to the project? Skip to the [Overview](#overview) section for a brief introduction.

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

To build the Docker image, run the following command:
```bash
docker build -t <image_name> .
```

To run the Docker container, run the following command:
```bash
docker run -it <image_name>
```

## Overview
"An important class of applications entails a robot monitoring, scrutinizing, or recording the evolution of an uncertain
time-extended process. This sort of situation leads to an interesting family
of planning problems in which the robot is limited in what it sees and must, thus, choose what to pay attention to. The distinguishing characteristic of this setting is that the robot has influence over what it  captures via its sensors, but exercises no causal authority over the evolving process. As such, the robot’s objective is to observe the underlying process and to produce a ‘chronicle’ of occurrent events, subject to a
goal specification of the sorts of event sequences that may be of interest.
This paper examines variants of such problems when the robot aims to
collect sets of observations to meet a rich specification of their sequential structure. We study this class of problems by modeling a stochastic
process via a variant of a hidden Markov model, and specify the event
sequences of interest as a regular language, developing a vocabulary of
‘mutators’ that enable sophisticated requirements to be expressed. Under different suppositions about the information gleaned about the event
model, we formulate and solve different planning problems. The core underlying idea is the construction of a product between the event model
and a specification automaton.

"A concrete motivation for this sort of setting, consider the
proliferation of home videos. These videos are, with remarkably few exceptions,
crummy specimens of the cinematic arts. They fail, generally, to establish and
then bracket a scene; they often founder in emphasizing the importance of key
subjects within the developing action, and are usually unsuccessful in attempts
to trace an evolving narrative arc. And the current generation of autonomous
personal robots and video drones, in their roles as costly and glorified ‘selfie
sticks,’ are set to follow suit. The trouble is that capturing footage to tell a story
is challenging. A camera can only record what you point it toward, so part of
the difficulty stems from the fact that you can’t know exactly how the scene will
unfold before it actually does. Moreover, what constitutes structure isn’t easily
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

## Running the code

The code has been containerized and MUST be run on Python 3.9. Libraries used are not compatible with other versions.

The current `Dockerfile` will run the code in the `main.py` file. It demonstrates all functionality of the codebase.

When running locally, the `main.ipynb` file contains the code to run the experiments. The code is well-documented and should be easy to follow.

## Questions?

If you have any questions, feel free to reach out to me at [tiernanlind@tamu.edu](mailto:tiernanlind@tamu.edu).

## Acknowledgements

Dylan A. Shell
Hazhar Rahmani
