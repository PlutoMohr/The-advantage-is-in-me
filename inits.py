# Game settings
TITLE = "SnakeGame"
GRID_SIZE = 30  # grid size
BLANK_SIZE = 40  # top size
ROWS = 10  # screen width
COLS = 10  # screen height
FPS = 10  # AI travel speed
FPS_human = 10  # human travel speed
FONT_NAME = 'arial'

# Define colors
WHITE = (255, 255, 255)
WHITE1 = (220, 220, 220)
WHITE2 = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (0, 155, 155)
BGCOLOR = BLACK
LINE_COLOR = BLUE1

INF = 1000000000

# AI settings
N_INPUT = 32
# N_HIDDEN1 = 20
N_HIDDEN1 = 24
# N_HIDDEN2 = 12
N_HIDDEN2 = 12
N_OUTPUT = 4
GENES_LEN = N_INPUT * N_HIDDEN1 + N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 * N_OUTPUT + N_HIDDEN1 + N_HIDDEN2 + N_OUTPUT
P_SIZE = 100  # Number of parents
C_SIZE = 400  # Number of children
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # down, up, left, right
MUTATE_RATE = 0.1  # mutation probability

n_state = 32
n_action = 4
weight_exc = 1
weight_inh = -0.5
trace_decay = 0.8
