"""
two agents, placing piece in turn, O or X

environment: 3x3 array as chess board

| X |   |   |
| X | O | O |
| X |   |   |

goal: win the game, have three continuous placement, horizontal, vertical, or diagonal

state: situation of 3x3 chess board, each slot is empty (None) or X or O

action: where to put next piece

reward: immediate value of a action

policy: the function, that action = function(state), here it's where-to-put-piece = policy(board-state)

value: expected long-term cumulative reward from a state or action.

two agents play in turn, and one agent don't receive immediate feedback after one action, its new state come after its opponent place its pice

maybe you are interested about afterstate
"""
