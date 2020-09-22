"""
Name: Israela Megira
ID: 209015817
"""

from pddlsim.local_simulator import LocalSimulator
from best_agent import best_agent
from Rmax import Rmax
import sys


# gets the domain and the problem from the arguments
#domain_path = sys.argv[2]
#problem_path = sys.argv[3]

""" deterministic domain and problem """
# maze
domain_path = "maze_domain.pddl"
problem_path = "maze_problem.pddl"

# freecell
domain_path = "freecell_domain.pddl"
problem_path = "freecell_problem.pddl"

# satellite
domain_path = "satellite_domain.pddl"
problem_path = "satellite_problem.pddl"

# rover
""" domain_path = "rover_domain (2).pddl"
problem_path = "rover_problem (2).pddl" """

""" NOT deterministic domain and problem """
# maze
domain_path = "maze_domain_multi_effect_food.pddl"
problem_path = "t_5_5_5_food-prob0.pddl"
#problem_path = "t_5_5_5_food-prob1.pddl"
#problem_path = "t_10_10_10.pddl"

# satellite
#domain_path = "satellite_domain_multi.pddl"
#problem_path = "satellite_problem_multi.pddl"

print LocalSimulator().run(sys.argv[2], sys.argv[3], best_agent())

