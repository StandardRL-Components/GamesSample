
# Generated: 2025-08-28T05:56:10.675121
# Source Brief: brief_02781.md
# Brief Index: 2781

        
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


# A small, embedded word list for the game.
# In a real application, this would be much larger.
WORD_LIST = {
    "ace", "act", "add", "ado", "ads", "aft", "age", "ago", "aha", "aid", "ail", "aim", "air", "ale", "all", "amp",
    "and", "ant", "any", "ape", "apt", "arc", "are", "ark", "arm", "art", "ash", "ask", "asp", "ate", "awe", "axe",
    "bad", "bag", "ban", "bar", "bat", "bay", "bed", "bee", "beg", "bet", "bid", "big", "bin", "bit", "boa", "bob",
    "bog", "boo", "bop", "bow", "box", "boy", "bud", "bug", "bum", "bun", "bus", "but", "buy", "bye", "cab", "cad",
    "cam", "can", "cap", "car", "cat", "chi", "con", "cop", "cot", "cow", "coy", "cry", "cub", "cud", "cue", "cup",
    "cut", "dab", "dad", "dam", "day", "den", "dew", "did", "die", "dig", "dim", "din", "dip", "doe", "dog", "don",
    "dot", "dry", "dub", "dud", "due", "dug", "dun", "duo", "dye", "ear", "eat", "ebb", "eel", "egg", "ego", "elf",
    "elm", "emu", "end", "eon", "era", "erg", "eta", "eve", "eye", "fad", "fan", "far", "fat", "fed", "fee", "fen",
    "few", "fib", "fig", "fin", "fir", "fit", "fix", "flu", "fly", "fob", "foe", "fog", "for", "fox", "fry", "fun",
    "fur", "gag", "gal", "gap", "gas", "gay", "gee", "gel", "gem", "get", "gig", "gin", "git", "gnu", "gob", "god",
    "goo", "got", "gum", "gun", "gut", "guy", "gym", "had", "hag", "ham", "has", "hat", "hay", "hem", "hen", "her",
    "hew", "hex", "hey", "hid", "hie", "him", "hip", "his", "hit", "hob", "hod", "hoe", "hog", "hop", "hot", "how",
    "hub", "hue", "hug", "huh", "hum", "hun", "hut", "ice", "icy", "igg", "ill", "imp", "ink", "inn", "ion", "ire",
    "irk", "ism", "its", "ivy", "jab", "jag", "jam", "jar", "jaw", "jay", "jet", "jew", "jib", "jig", "job", "jog",
    "jot", "joy", "jug", "jun", "jus", "jut", "keg", "ken", "key", "kid", "kin", "kip", "kit", "lab", "lac", "lad",
    "lag", "lam", "lap", "law", "lax", "lay", "lea", "led", "lee", "leg", "let", "lib", "lid", "lie", "lip", "lit",
    "lob", "log", "lop", "lot", "low", "lox", "lug", "lye", "mad", "man", "map", "mar", "mat", "maw", "may", "men",
    "met", "mew", "mid", "mil", "mix", "mob", "mod", "mol", "mom", "mon", "mop", "mot", "mow", "mud", "mug", "mum",
    "nab", "nag", "nap", "nay", "neb", "nee", "net", "new", "nib", "nil", "nip", "nit", "nix", "nob", "nod", "nog",
    "noh", "nom", "noo", "nor", "not", "now", "nub", "nun", "nut", "oaf", "oak", "oar", "oat", "odd", "ode", "off",
    "oft", "ohm", "oho", "oil", "old", "one", "orb", "ore", "our", "out", "ova", "owe", "owl", "own", "pad", "pal",
    "pan", "pap", "par", "pat", "paw", "pay", "pea", "peg", "pen", "pep", "per", "pet", "pew", "phi", "pie", "pig",
    "pin", "pip", "pit", "ply", "pod", "poi", "pol", "pop", "pot", "pro", "psi", "pub", "pud", "pug", "pun", "pup",
    "pus", "put", "qua", "rad", "rag", "rah", "ram", "ran", "rap", "rat", "raw", "ray", "red", "ref", "reg", "rem",
    "rep", "rev", "rho", "rib", "rid", "rig", "rim", "rip", "rob", "rod", "roe", "rot", "row", "rub", "rue", "rug",
    "rum", "run", "rut", "rye", "sab", "sac", "sad", "sag", "sap", "sat", "saw", "sax", "say", "sea", "sec", "see",
    "sen", "set", "sew", "sex", "she", "shh", "shy", "sin", "sip", "sir", "sis", "sit", "six", "ski", "sky", "sly",
    "sob", "sod", "sol", "son", "sop", "sot", "sow", "soy", "spa", "spy", "sty", "sub", "sue", "sum", "sun", "sup",
    "tab", "tad", "tag", "tam", "tan", "tap", "tar", "tat", "tau", "tax", "tea", "ted", "tee", "ten", "the", "tho",
    "thy", "tic", "tie", "til", "tin", "tip", "tit", "toe", "tog", "tom", "ton", "too", "top", "tor", "tot", "tow",
    "toy", "try", "tub", "tug", "tux", "two", "use", "van", "vat", "vet", "vie", "vim", "wad", "wag", "wan", "war",
    "was", "wax", "way", "web", "wed", "wee", "wet", "who", "why", "wig", "win", "wit", "woe", "wok", "won", "woo",
    "wow", "wry", "wye", "yacht", "zealot", "zero", "zest", "zinc", "zing", "zipped", "zone", "zoom",
    "word", "words", "game", "play", "player", "score", "time", "timer", "grid", "letter", "select", "move", "find",
    "create", "puzzle", "challenge", "arcade", "fast", "paced", "corner"
}
PREFIX_SET = {word[:i] for word in WORD_LIST for i in range(1, len(word) + 1)}

# English letter frequencies for more realistic grid generation
LETTER_FREQUENCIES = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99,
    'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
    'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
}
LETTERS, WEIGHTS = zip(*LETTER_FREQUENCIES.items())

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select adjacent letters to form a word. "
        "Press shift to undo the last letter selection. Formed words are submitted automatically."
    )

    game_description = (
        "Find and form words from a grid of letters against the clock. "
        "Score points by creating words, with bonuses for longer ones. Reach 500 points to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.fps = 30

        # Game constants
        self.grid_size = 10
        self.win_score = 500
        self.time_limit_seconds = 60
        self.max_steps = self.time_limit_seconds * self.fps

        # Visuals
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        self.COLOR_BG = (15, 23, 42)
        self.COLOR_GRID = (30, 41, 59)
        self.COLOR_LETTER = (226, 232, 240)
        self.COLOR_CURSOR = (56, 189, 248)
        self.COLOR_PATH = (250, 204, 21)
        self.COLOR_PATH_LINE = (234, 179, 8, 200)
        self.COLOR_VALID_MOVE = (34, 197, 94, 64)
        self.COLOR_UI_TEXT = (241, 245, 249)
        self.COLOR_TIMER_WARN = (239, 68, 68)

        self.tile_size = 36
        self.grid_width = self.grid_size * self.tile_size
        self.grid_height = self.grid_size * self.tile_size
        self.grid_x_offset = (self.screen_width - self.grid_width) // 2
        self.grid_y_offset = (self.screen_height - self.grid_height) // 2

        # Game state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.current_path = []
        self.current_word = ""
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        self.word_pop_effects = []

        # Input handling
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        self.move_cooldown_max = 4 # frames

        # Initialize state
        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.timer = self.max_steps
        self._generate_grid()
        
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        self.visual_cursor_pos = [float(self.cursor_pos[0]), float(self.cursor_pos[1])]
        
        self.current_path = []
        self.current_word = ""
        self.particles = []
        self.word_pop_effects = []

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.timer <= 0 or self.score >= self.win_score
        if self.game_over:
            if self.score >= self.win_score and not self.win_state:
                reward = 100
                self.win_state = True
            elif self.timer <= 0 and self.score < self.win_score:
                reward = -10
            
            return (self._get_observation(), reward, True, False, self._get_info())

        # Unpack action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # --- Game Logic ---
        self.timer -= 1
        self.steps += 1
        
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # Handle movement
        if movement > 0 and self.move_cooldown == 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.grid_size - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.grid_size - 1)
            self.move_cooldown = self.move_cooldown_max

        # Handle actions (on press)
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if space_press:
            reward += self._handle_selection()

        if shift_press:
            self._handle_deselection()

        self._update_particles()
        self._update_animations()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        terminated = self.timer <= 0 or self.score >= self.win_score
        
        return (self._get_observation(), reward, terminated, False, self._get_info())

    def _handle_selection(self):
        selected_pos = tuple(self.cursor_pos)
        
        # Check if selection is valid
        is_adjacent = not self.current_path or self._is_adjacent(selected_pos, self.current_path[-1])
        is_new = selected_pos not in self.current_path

        if is_adjacent and is_new:
            self.current_path.append(selected_pos)
            self.current_word += self.grid[selected_pos[1]][selected_pos[0]]
            
            # Check for word submission
            if len(self.current_word) >= 3 and self.current_word.lower() in WORD_LIST:
                # Sound: Word_Success.wav
                word_score = len(self.current_word)
                bonus = 5 if len(self.current_word) > 5 else 0
                self.score += word_score + bonus
                
                # Add time back
                self.timer = min(self.max_steps, self.timer + word_score * self.fps // 2)

                self._create_word_pop_effect()
                self._create_particles_for_path()
                
                self.current_path = []
                self.current_word = ""
                return word_score + bonus

            # Reward for valid prefix
            if self.current_word.lower() in PREFIX_SET:
                return 0.1
            else: # Invalid prefix, reset path
                # Sound: Error.wav
                self.current_path = []
                self.current_word = ""
                return -0.1
        else: # Invalid selection (not adjacent or already used), reset path
            # Sound: Error.wav
            self.current_path = []
            self.current_word = ""
            return -0.1

    def _handle_deselection(self):
        if self.current_path:
            # Sound: Backspace.wav
            self.current_path.pop()
            self.current_word = self.current_word[:-1]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid and Letters ---
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = self._get_tile_rect(x, y)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=3)
                letter = self.grid[y][x]
                
                size_multiplier = 1.0
                for pop_effect in self.word_pop_effects:
                    if pop_effect["pos"] == (x,y):
                        size_multiplier = pop_effect["scale"]

                font_size = int(self.tile_size * 0.7 * size_multiplier)
                if font_size <= 1: continue
                
                temp_font = pygame.font.Font(None, font_size)
                letter_surf = temp_font.render(letter, True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=rect.center)
                self.screen.blit(letter_surf, letter_rect)

        # --- Highlight Potential Moves ---
        if self.current_path:
            last_pos = self.current_path[-1]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    adj_pos = (last_pos[0] + dx, last_pos[1] + dy)
                    if 0 <= adj_pos[0] < self.grid_size and 0 <= adj_pos[1] < self.grid_size and adj_pos not in self.current_path:
                        rect = self._get_tile_rect(adj_pos[0], adj_pos[1])
                        s = pygame.Surface(rect.size, pygame.SRCALPHA)
                        s.fill(self.COLOR_VALID_MOVE)
                        self.screen.blit(s, rect.topleft)

        # --- Highlight Current Path ---
        if self.current_path:
            points = []
            for i, pos in enumerate(self.current_path):
                rect = self._get_tile_rect(pos[0], pos[1])
                pygame.gfxdraw.box(self.screen, rect, self.COLOR_PATH + (100,))
                points.append(rect.center)
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH_LINE, False, points, 4)
            for p in points:
                pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 5, self.COLOR_PATH)
                pygame.gfxdraw.aacircle(self.screen, p[0], p[1], 5, self.COLOR_PATH)

        # --- Draw Cursor ---
        lerp_rate = 0.4
        self.visual_cursor_pos[0] += (self.cursor_pos[0] - self.visual_cursor_pos[0]) * lerp_rate
        self.visual_cursor_pos[1] += (self.cursor_pos[1] - self.visual_cursor_pos[1]) * lerp_rate
        
        cursor_rect = self._get_tile_rect(self.visual_cursor_pos[0], self.visual_cursor_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # --- Draw Particles ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 15))

        # Timer
        time_left = max(0, self.timer / self.fps)
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_UI_TEXT
        timer_surf = self.font_medium.render(f"Time: {time_left:.1f}", True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.screen_width - 20, 15))
        self.screen.blit(timer_surf, timer_rect)

        # Current Word
        display_word = self.current_word if self.current_word else "_"
        word_surf = self.font_large.render(display_word, True, self.COLOR_PATH)
        word_rect = word_surf.get_rect(center=(self.screen_width / 2, self.screen_height - 30))
        self.screen.blit(word_surf, word_rect)
        
        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.win_score else "TIME'S UP!"
            msg_surf = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            final_score_surf = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 30))
            self.screen.blit(final_score_surf, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.fps),
            "current_word": self.current_word,
        }

    def _generate_grid(self):
        self.grid = self.np_random.choice(
            LETTERS, 
            size=(self.grid_size, self.grid_size), 
            p=[w / sum(WEIGHTS) for w in WEIGHTS]
        ).tolist()

    def _get_tile_rect(self, x, y):
        return pygame.Rect(
            self.grid_x_offset + x * self.tile_size,
            self.grid_y_offset + y * self.tile_size,
            self.tile_size,
            self.tile_size
        )

    def _is_adjacent(self, pos1, pos2):
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) == 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["radius"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["radius"] -= 0.2

    def _create_particles_for_path(self):
        for pos in self.current_path:
            center = self._get_tile_rect(pos[0], pos[1]).center
            for _ in range(10):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                self.particles.append({
                    "pos": list(center),
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "radius": self.np_random.uniform(3, 7),
                    "color": random.choice([self.COLOR_PATH, (253, 230, 138), (252, 211, 77)])
                })
    
    def _create_word_pop_effect(self):
        for pos in self.current_path:
            self.word_pop_effects.append({"pos": pos, "scale": 1.5, "decay": 0.05})

    def _update_animations(self):
        self.word_pop_effects = [e for e in self.word_pop_effects if e["scale"] > 1.0]
        for effect in self.word_pop_effects:
            effect["scale"] -= effect["decay"]
            effect["decay"] *= 1.1 # accelerate decay

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Word Grid")
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose and blit the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.fps)
        
    env.close()