CHARACTERS_MAP = {"wall": "@",
                  "space": " ",
                  "apple": "A",
                  "beam": "F",
                  "sight": "S",
                  "agent": "1"}

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [0, 0, 0],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player
                   'S': [150, 150, 150],  # sight of agent

                   # Colours for agents. R value is a unique identifier
                   '1': [255, 0, 0],  # Red
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow
SPAWN_PROB = [0, 0.01, 0.01, 0.05, 0.05, 0.1]
SMALL_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A    A     A   PA     A  @',
    '@ AAA  AAA P AAA  AAA P AAA @',
    '@  AP   A     A    A     A  @',
    '@          P         P      @',
    '@P   A     A    A    A     P@',
    '@   AAAP  AAA  AAA  AAAP    @',
    '@    A   P A    A   PA      @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']
