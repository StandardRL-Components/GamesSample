
# Generated: 2025-08-27T18:46:26.559506
# Source Brief: brief_01946.md
# Brief Index: 1946

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move cursor. Press space to select a letter. "
        "Hold shift to submit the word."
    )

    game_description = (
        "Connect adjacent letters in a grid to form words and score points "
        "before time runs out. Find 5 words to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 6000
    WORDS_TO_WIN = 5
    GRID_DIM = 6
    
    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 65, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_RARE = (255, 215, 0) # Gold
    COLOR_CURSOR = (255, 100, 100, 150)
    COLOR_PATH = (100, 150, 255)
    COLOR_VALID = (100, 255, 100)
    COLOR_INVALID = (255, 80, 80)
    COLOR_UI_BG = (35, 50, 65)
    COLOR_UI_ACCENT = (255, 165, 0)
    
    # --- Word Data ---
    # A small, embedded dictionary for portability
    WORD_LIST = {
        "ace", "act", "add", "ado", "ads", "aft", "age", "ago", "aid", "ail",
        "aim", "air", "ale", "all", "amp", "and", "ant", "any", "ape", "apt",
        "arc", "are", "ark", "arm", "art", "ash", "ask", "asp", "ate", "awe",
        "axe", "bad", "bag", "ban", "bar", "bat", "bed", "bee", "beg", "bet",
        "bid", "big", "bin", "bit", "boa", "bob", "bog", "boo", "bop", "bow",
        "box", "boy", "bra", "bud", "bug", "bum", "bun", "bus", "but", "buy",
        "bye", "cab", "cad", "cam", "can", "cap", "car", "cat", "cod", "cog",
        "con", "coo", "cop", "cot", "cow", "coy", "cry", "cub", "cud", "cue",
        "cup", "cur", "cut", "dab", "dad", "dam", "day", "den", "dew", "did",
        "die", "dig", "dim", "din", "dip", "doc", "doe", "dog", "don", "dot",
        "dry", "dub", "dud", "due", "dug", "dun", "duo", "dye", "ear", "eat",
        "ebb", "eel", "egg", "ego", "eke", "elf", "elk", "elm", "emu", "end",
        "era", "erg", "eta", "eve", "eye", "fad", "fan", "far", "fat", "fed",
        "fee", "fen", "few", "fib", "fig", "fin", "fir", "fit", "fix", "flu",
        "fly", "foe", "fog", "for", "fox", "fry", "fun", "fur", "gag", "gal",
        "gap", "gas", "gay", "gee", "gel", "gem", "get", "gig", "gin", "god",
        "goo", "got", "gum", "gun", "gut", "guy", "gym", "had", "ham", "has",
        "hat", "hay", "hem", "hen", "her", "hew", "hex", "hey", "hid", "him",
        "hip", "his", "hit", "hoe", "hog", "hop", "hot", "how", "hub", "hue",
        "hug", "huh", "hum", "hut", "ice", "icy", "igg", "ill", "imp", "ink",
        "inn", "ion", "ire", "irk", "its", "ivy", "jab", "jag", "jam", "jar",
        "jaw", "jay", "jet", "jig", "job", "jog", "jot", "joy", "jug", "jut",
        "keg", "ken", "key", "kid", "kin", "kit", "lab", "lac", "lad", "lag",
        "lam", "lap", "law", "lay", "lea", "led", "lee", "leg", "let", "lib",
        "lid", "lie", "lip", "lit", "lob", "log", "lop", "lot", "low", "lug",
        "mad", "man", "map", "mar", "mat", "maw", "may", "men", "met", "mew",
        "mid", "mil", "mob", "mod", "mol", "mom", "mon", "moo", "mop", "mow",
        "mud", "mug", "mum", "nab", "nag", "nap", "nay", "neb", "net", "new",
        "nib", "nil", "nip", "nit", "nix", "nob", "nod", "nor", "not", "now",
        "nun", "nut", "oaf", "oak", "oar", "oat", "odd", "ode", "off", "oft",
        "ohm", "oho", "oil", "old", "one", "orb", "ore", "our", "out", "owe",
        "owl", "own", "pad", "pal", "pan", "pap", "par", "pat", "paw", "pay",
        "pea", "peg", "pen", "pep", "per", "pet", "pew", "phi", "pie", "pig",
        "pin", "pip", "pit", "ply", "pod", "poi", "pol", "pop", "pot", "pro",
        "pry", "pub", "pug", "pun", "pup", "pus", "put", "qua", "rad", "rag",
        "ram", "ran", "rap", "rat", "raw", "ray", "red", "ref", "reg", "rem",
        "rep", "rev", "rib", "rid", "rig", "rim", "rip", "rob", "rod", "roe",
        "rot", "row", "rub", "rue", "rug", "rum", "run", "rut", "rye", "sab",
        "sac", "sad", "sag", "sap", "sat", "saw", "say", "sea", "sec", "see",
        "sen", "set", "sew", "sex", "she", "shy", "sib", "sic", "sin", "sip",
        "sir", "sis", "sit", "six", "ski", "sky", "sly", "sob", "sod", "sol",
        "son", "sop", "sow", "soy", "spa", "spy", "sty", "sub", "sue", "sum",
        "sun", "sup", "tab", "tad", "tag", "tam", "tan", "tap", "tar", "tat",
        "tax", "tea", "ted", "tee", "ten", "the", "tho", "thy", "tic", "tie",
        "til", "tin", "tip", "tit", "toe", "tog", "tom", "ton", "too", "top",
        "tor", "tot", "tow", "toy", "try", "tub", "tug", "tui", "tun", "two",
        "ugh", "ump", "urn", "use", "van", "vat", "vet", "vie", "vig", "vim",
        "vow", "wad", "wag", "wan", "war", "was", "wax", "way", "web", "wed",
        "wee", "wet", "who", "why", "wig", "win", "wit", "woe", "wok", "won",
        "woo", "wow", "wry", "wye", "yacht", "yack", "yam", "yap", "yard", "yarn",
        "yaw", "yawn", "yea", "yeah", "year", "yell", "yelp", "yen", "yep", "yes",
        "yet", "yield", "yoke", "yolk", "yonder", "yore", "you", "your", "yours",
        "youth", "yowl", "yule", "zany", "zap", "zapped", "zaps", "zeal",
        "zealot", "zebra", "zebu", "zed", "zeds", "zee", "zees", "zenith",
        "zero", "zeros", "zest", "zesty", "zeta", "zig", "zigzag", "zillion",
        "zinc", "zing", "zip", "zipped", "zipper", "zips", "zit", "ziti", "zits",
        "zodiac", "zombie", "zone", "zoned", "zones", "zoning", "zoo", "zoom",
        "zoomed", "zooms"
    }
    
    LETTER_FREQUENCIES = (
        "E" * 12 + "T" * 9 + "A" * 8 + "O" * 8 + "I" * 7 + "N" * 7 +
        "S" * 6 + "H" * 6 + "R" * 6 + "D" * 4 + "L" * 4 + "C" * 3 +
        "U" * 3 + "M" * 3 + "W" * 2 + "F" * 2 + "G" * 2 + "Y" * 2 +
        "P" * 2 + "B" * 2 + "V" * 1 + "K" * 1 + "J" * 1 + "X" * 1 +
        "Q" * 1 + "Z" * 1
    )
    RARE_LETTERS = {'J', 'K', 'Q', 'X', 'Z'}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_grid = pygame.font.Font(None, 48)
        self.font_ui_large = pygame.font.Font(None, 36)
        self.font_ui_medium = pygame.font.Font(None, 28)
        self.font_feedback = pygame.font.Font(None, 32)

        self.grid_surface = pygame.Surface((self.SCREEN_HEIGHT, self.SCREEN_HEIGHT))
        self.cell_size = self.SCREEN_HEIGHT // self.GRID_DIM
        self.grid_offset_x = (self.SCREEN_WIDTH - self.SCREEN_HEIGHT) // 2
        self.grid_offset_y = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.grid = self._generate_grid()
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.current_word_path = []
        self.current_word_str = ""
        self.found_words = set()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.feedback_animations = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Actions ---
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._handle_selection()
            
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            reward += self._handle_submission()
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.steps += 1
        
        # --- Update animations ---
        self._update_animations()

        # --- Check Termination ---
        terminated = False
        if len(self.found_words) >= self.WORDS_TO_WIN:
            reward += 50
            terminated = True
            self._add_feedback_animation("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_VALID, 60)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self._add_feedback_animation("TIME'S UP!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_INVALID, 60)
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_grid(self):
        return [
            [random.choice(self.LETTER_FREQUENCIES) for _ in range(self.GRID_DIM)]
            for _ in range(self.GRID_DIM)
        ]

    def _handle_movement(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        self.cursor_pos[0] = np.clip(x, 0, self.GRID_DIM - 1)
        self.cursor_pos[1] = np.clip(y, 0, self.GRID_DIM - 1)
    
    def _handle_selection(self):
        pos = tuple(self.cursor_pos)
        
        if pos in self.current_word_path:
            # Deselect last letter if re-clicked
            if len(self.current_word_path) > 0 and pos == self.current_word_path[-1]:
                self.current_word_path.pop()
                self.current_word_str = self.current_word_str[:-1]
            return 0

        is_adjacent = False
        if not self.current_word_path:
            is_adjacent = True
        else:
            last_pos = self.current_word_path[-1]
            if abs(pos[0] - last_pos[0]) <= 1 and abs(pos[1] - last_pos[1]) <= 1:
                is_adjacent = True

        if is_adjacent:
            letter = self.grid[pos[1]][pos[0]]
            self.current_word_path.append(pos)
            self.current_word_str += letter
            # sfx: select_letter.wav
            return 0.1 if letter in self.RARE_LETTERS else -0.02
        else:
            # sfx: invalid_selection.wav
            return -0.1 # Penalty for trying to select non-adjacent

    def _handle_submission(self):
        word = self.current_word_str.lower()
        reward = 0
        
        is_valid = (
            len(word) >= 3 and
            word in self.WORD_LIST and
            word not in self.found_words
        )
        
        if is_valid:
            self.found_words.add(word)
            word_reward = len(word)
            self.score += word_reward
            reward += word_reward
            # sfx: valid_word.wav
            
            for i, pos in enumerate(self.current_word_path):
                center_x = self.grid_offset_x + int((pos[0] + 0.5) * self.cell_size)
                center_y = int((pos[1] + 0.5) * self.cell_size)
                self._add_feedback_animation(f"+{word_reward}", (center_x, center_y), self.COLOR_VALID, 30, i * 5, True)
        else:
            reward -= 0.5 # Penalty for bad submission
            # sfx: invalid_word.wav
            for i, pos in enumerate(self.current_word_path):
                center_x = self.grid_offset_x + int((pos[0] + 0.5) * self.cell_size)
                center_y = int((pos[1] + 0.5) * self.cell_size)
                self._add_feedback_animation("X", (center_x, center_y), self.COLOR_INVALID, 20, i * 3)

        self.current_word_path = []
        self.current_word_str = ""
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and letters
        self.grid_surface.fill(self.COLOR_BG)
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.grid_surface, self.COLOR_GRID, rect, 1)
                
                letter = self.grid[y][x]
                color = self.COLOR_RARE if letter in self.RARE_LETTERS else self.COLOR_TEXT
                text_surf = self.font_grid.render(letter, True, color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.grid_surface.blit(text_surf, text_rect)
        
        # Draw selection path
        if len(self.current_word_path) > 1:
            points = [
                (int((pos[0] + 0.5) * self.cell_size), int((pos[1] + 0.5) * self.cell_size))
                for pos in self.current_word_path
            ]
            pygame.draw.lines(self.grid_surface, self.COLOR_PATH, False, points, 5)

        # Draw circles on selected letters
        for pos in self.current_word_path:
            center = (int((pos[0] + 0.5) * self.cell_size), int((pos[1] + 0.5) * self.cell_size))
            pygame.gfxdraw.aacircle(self.grid_surface, center[0], center[1], self.cell_size // 3, self.COLOR_PATH)
            pygame.gfxdraw.filled_circle(self.grid_surface, center[0], center[1], self.cell_size // 3, self.COLOR_PATH)


        # Draw cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.cell_size,
            self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        cursor_surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        cursor_surf.fill(self.COLOR_CURSOR)
        self.grid_surface.blit(cursor_surf, cursor_rect.topleft)

        self.screen.blit(self.grid_surface, (self.grid_offset_x, self.grid_offset_y))
        
        # Draw feedback animations on top of everything else
        for anim in self.feedback_animations:
            if anim['delay'] <= 0:
                font = self.font_ui_large if "WIN" in anim['text'] or "UP" in anim['text'] else self.font_feedback
                text_surf = font.render(anim['text'], True, anim['color'])
                alpha = int(255 * (anim['timer'] / anim['duration']))
                text_surf.set_alpha(alpha)
                
                pos = anim['pos']
                if anim['float']:
                    pos = (pos[0], pos[1] - (1 - (anim['timer'] / anim['duration'])) * 20)

                text_rect = text_surf.get_rect(center=pos)
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Top Bar ---
        top_bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, top_bar_rect)
        
        # Words Found
        words_text = f"WORDS: {len(self.found_words)} / {self.WORDS_TO_WIN}"
        words_surf = self.font_ui_medium.render(words_text, True, self.COLOR_TEXT)
        self.screen.blit(words_surf, (15, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=20)
        self.screen.blit(score_surf, score_rect)
        
        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"TIME: {time_left}"
        time_surf = self.font_ui_medium.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(right=self.SCREEN_WIDTH - 15, centery=20)
        self.screen.blit(time_surf, time_rect)

        # --- Bottom Bar (Current Word) ---
        bottom_bar_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bottom_bar_rect)
        
        word_surf = self.font_ui_large.render(self.current_word_str, True, self.COLOR_UI_ACCENT)
        word_rect = word_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT - 20)
        self.screen.blit(word_surf, word_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "current_word": self.current_word_str
        }

    def _add_feedback_animation(self, text, pos, color, duration, delay=0, float_up=False):
        self.feedback_animations.append({
            'text': text, 'pos': pos, 'color': color, 'duration': duration,
            'timer': duration, 'delay': delay, 'float': float_up
        })
    
    def _update_animations(self):
        for anim in self.feedback_animations[:]:
            if anim['delay'] > 0:
                anim['delay'] -= 1
            else:
                anim['timer'] -= 1
                if anim['timer'] <= 0:
                    self.feedback_animations.remove(anim)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Word Grid Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement = 0 # no-op
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Use get_pressed for held keys
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)

        # Since auto_advance is False, we control the step rate
        pygame.time.Clock().tick(30) # Limit to 30 FPS for human playability
        
    pygame.quit()