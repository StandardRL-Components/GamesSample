
# Generated: 2025-08-27T23:10:13.146177
# Source Brief: brief_03366.md
# Brief Index: 3366

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a letter. Press Shift to submit the current word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect adjacent letters on the grid to form words against the clock. Form 5 words to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Game Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    CELL_SIZE = 40
    GRID_TOP_MARGIN = 100
    GRID_LEFT_MARGIN = (640 - GRID_WIDTH * CELL_SIZE) // 2
    FPS = 30
    GAME_DURATION_SECONDS = 60
    WIN_CONDITION_WORDS = 5
    MAX_STEPS = FPS * GAME_DURATION_SECONDS + 10  # A bit of buffer

    # --- Colors ---
    COLOR_BG = (25, 28, 44)
    COLOR_GRID_LINE = (55, 58, 74)
    COLOR_UI_BG = (40, 43, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_LETTER_DEFAULT = (200, 200, 210)
    COLOR_CURSOR = (0, 190, 255)
    COLOR_SELECTED_PATH = (255, 215, 0)
    COLOR_FEEDBACK_GOOD = (90, 255, 150)
    COLOR_FEEDBACK_BAD = (255, 90, 90)
    COLOR_TIMER_LOW = (255, 100, 100)

    # --- Word List ---
    WORD_LIST = {
        "cat", "dog", "sun", "run", "big", "red", "bed", "egg", "fun", "get", "hat", "jet",
        "map", "pen", "rat", "sit", "top", "win", "zoo", "ace", "ale", "ape", "arm", "art",
        "ate", "awe", "axe", "bag", "ban", "bat", "bee", "bet", "bid", "bin", "bit", "boa",
        "bob", "bog", "bot", "bow", "box", "boy", "bud", "bug", "bum", "bun", "bus", "but",
        "buy", "bye", "cab", "cad", "cam", "can", "cap", "car", "cob", "cod", "cog", "con",
        "cop", "cot", "cow", "coy", "cry", "cub", "cud", "cup", "cur", "cut", "dab", "dad",
        "dam", "dan", "day", "den", "dew", "did", "die", "dig", "dim", "din", "dip", "doc",
        "doe", "dot", "dry", "dub", "dud", "due", "dug", "dun", "duo", "dye", "ear", "eat",
        "eel", "elf", "elk", "elm", "end", "eon", "era", "erg", "eta", "eve", "eye", "fad",
        "fag", "fan", "far", "fat", "fed", "fee", "fen", "few", "fez", "fib", "fig", "fin",
        "fir", "fit", "fix", "fly", "fob", "foe", "fog", "fop", "for", "fox", "fro", "fry",
        "fug", "fun", "fur", "gab", "gad", "gag", "gal", "gam", "gap", "gas", "gay", "gel",
        "gem", "get", "gig", "gin", "gip", "git", "gnu", "gob", "god", "goo", "got", "gum",
        "gun", "gut", "guy", "gym", "gyp", "had", "hag", "hah", "ham", "has", "hat", "hay",
        "hem", "hen", "her", "hes", "hew", "hex", "hey", "hid", "hie", "him", "hin", "hip",
        "his", "hit", "hob", "hod", "hoe", "hog", "hon", "hop", "hot", "how", "hub", "hue",
        "hug", "huh", "hum", "hun", "hut", "ice", "icy", "igg", "ill", "imp", "ink", "inn",
        "ion", "ire", "irk", "its", "ivy", "jab", "jag", "jam", "jar", "jaw", "jay", "jet",
        "jib", "jig", "jin", "job", "joe", "jog", "jot", "jow", "joy", "jug", "jun", "jus",
        "jut", "keg", "ken", "key", "kid", "kin", "kip", "kit", "lab", "lac", "lad", "lag",
        "lam", "lap", "lar", "las", "lat", "lav", "law", "lax", "lay", "lea", "led", "lee",
        "leg", "lei", "lek", "let", "leu", "lev", "lex", "ley", "lib", "lid", "lie", "lin",
        "lip", "lis", "lit", "lob", "log", "loo", "lop", "lot", "low", "lox", "lug", "lye",
        "mac", "mad", "mae", "mag", "man", "map", "mar", "mas", "mat", "maw", "max", "may",
        "med", "meg", "mel", "men", "met", "mew", "mid", "mig", "mil", "mim", "mir", "mix",
        "moa", "mob", "moc", "mod", "mog", "mol", "mom", "mon", "moo", "mop", "mor", "mos",
        "mot", "mow", "mud", "mug", "mum", "mun", "mus", "mut", "myc", "nab", "nag", "nah",
        "nam", "nan", "nap", "nat", "nay", "neb", "nee", "neg", "net", "new", "nib", "nil",
        "nim", "nip", "nit", "nix", "nob", "nod", "nog", "noh", "nom", "noo", "nor", "nos",
        "not", "now", "nub", "nun", "nus", "nut", "oaf", "oak", "oar", "oat", "obi", "odd",
        "ode", "off", "oft", "ohm", "oho", "ohs", "oil", "oka", "oke", "old", "ole", "oma",
        "one", "ono", "ons", "ooh", "oot", "ope", "ops", "opt", "ora", "orb", "orc", "ore",
        "ors", "ort", "ose", "oud", "our", "out", "ova", "owe", "owl", "own", "oxo", "oxy",
        "pac", "pad", "pal", "pam", "pan", "pap", "par", "pas", "pat", "paw", "pax", "pay",
        "pea", "pec", "ped", "pee", "peg", "peh", "pen", "pep", "per", "pes", "pet", "pew",
        "phi", "pho", "pht", "pia", "pic", "pie", "pig", "pin", "pip", "pis", "pit", "piu",
        "pix", "ply", "pod", "poi", "pol", "pom", "pop", "pot", "pow", "pox", "pro", "pry",
        "psi", "pub", "pud", "pug", "pun", "pup", "pur", "pus", "put", "pya", "pye", "pyx",
        "qat", "qua", "rad", "rag", "rah", "rai", "raj", "ram", "ran", "rap", "ras", "rat",
        "raw", "rax", "ray", "reb", "rec", "red", "ree", "ref", "reg", "rei", "rem", "rep",
        "res", "ret", "rev", "rex", "rho", "ria", "rib", "rid", "rif", "rig", "rim", "rin",
        "rip", "rob", "roc", "rod", "roe", "rom", "rot", "row", "rub", "rue", "rug", "rum",
        "run", "rut", "rye", "sab", "sac", "sad", "sae", "sag", "sal", "sap", "sat", "sau",
        "saw", "sax", "say", "sea", "sec", "see", "seg", "sei", "sel", "sen", "ser", "set",
        "sew", "sex", "sha", "she", "shh", "sho", "shy", "sib", "sic", "sim", "sin", "sip",
        "sir", "sis", "sit", "six", "ski", "sky", "sly", "sob", "sod", "sol", "som", "son",
        "sop", "sos", "sot", "sou", "sow", "sox", "soy", "spa", "spy", "sri", "sty", "sub",
        "sue", "suk", "sum", "sun", "sup", "suq", "syn", "tab", "tad", "tae", "tag", "taj",
        "tam", "tan", "tao", "tap", "tar", "tas", "tat", "tau", "tav", "taw", "tax", "tea",
        "ted", "tee", "teg", "tel", "ten", "tes", "tet", "tew", "the", "tho", "thy", "tic",
        "tie", "til", "tin", "tip", "tis", "tit", "tod", "toe", "tog", "tom", "ton", "too",
        "top", "tor", "tot", "tow", "toy", "try", "tsk", "tub", "tug", "tui", "tun", "tup",
        "tut", "tux", "twa", "two", "tye", "udo", "ugh", "uke", "ulu", "umm", "ump", "uns",
        "upo", "ups", "urb", "urd", "urn", "urp", "use", "uta", "ute", "uts", "vac", "van",
        "var", "vas", "vat", "vau", "vav", "vaw", "vee", "veg", "vet", "vex", "via", "vid",
        "vie", "vig", "vim", "vin", "vis", "voe", "vog", "vow", "vox", "vug", "vum", "wab",
        "wad", "wae", "wag", "wan", "wap", "war", "was", "wat", "waw", "wax", "way", "web",
        "wed", "wee", "wen", "wet", "wha", "who", "why", "wig", "win", "wis", "wit", "wiz",
        "woe", "wog", "wok", "won", "woo", "wop", "wos", "wot", "wow", "wry", "wud", "wye",
        "wyn", "xis", "yag", "yah", "yak", "yam", "yap", "yar", "yaw", "yay", "yea", "yeh",
        "yen", "yep", "yes", "yet", "yew", "yid", "yin", "yip", "yob", "yod", "yok", "yom",
        "yon", "you", "yow", "yuk", "yum", "yup", "zag", "zap", "zas", "zax", "zed", "zee",
        "zek", "zen", "zep", "zig", "zin", "zip", "zit", "zoa", "zoo", "zos", "zuz", "word",
        "game", "play", "code", "grid", "time", "find", "make", "five", "good", "best",
        "fast", "slow", "line", "path", "form", "list", "test", "core", "cell", "move",
        "next", "last", "long", "term", "view", "text", "font", "draw", "loop", "step",
        "quit", "exit", "win", "lose", "over", "full", "part", "blue", "red", "gold",
        "python", "arcade", "letter", "visual", "action", "reward", "state", "agent",
        "expert", "puzzle", "player", "design", "create", "select", "submit", "valid",
        "connect", "quality", "minimal", "render", "cursor", "feedback", "bright", "clean"
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_feedback = pygame.font.SysFont("Arial", 28, bold=True)

        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.formed_words = []
        self.score = 0
        self.time_remaining = 0
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.word_feedback = {"text": "", "color": (0,0,0), "alpha": 0}

        self.reset()
        # self.validate_implementation() # Optional: uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_path = []
        self.formed_words = []
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.word_feedback = {"text": "", "color": (0,0,0), "alpha": 0}

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.auto_advance:
            dt = self.clock.tick(self.FPS) / 1000.0
            self.time_remaining -= dt
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.GRID_WIDTH
        self.cursor_pos[1] %= self.GRID_HEIGHT

        # 2. Handle letter selection (Space press)
        if space_pressed:
            pos = tuple(self.cursor_pos)
            if pos in self.selected_path:
                # Deselect if it's the last selected letter
                if pos == self.selected_path[-1]:
                    self.selected_path.pop()
                    # No specific reward for deselecting
            else:
                is_valid_selection = False
                if not self.selected_path:
                    is_valid_selection = True
                else:
                    last_pos = self.selected_path[-1]
                    if self._is_adjacent(pos, last_pos):
                        is_valid_selection = True
                
                if is_valid_selection:
                    self.selected_path.append(pos)
                    reward += 0.1
                    self._create_particles(self._cell_to_pixel(pos), self.COLOR_FEEDBACK_GOOD, 5)
                else:
                    reward -= 0.1
                    self._create_particles(self._cell_to_pixel(pos), self.COLOR_FEEDBACK_BAD, 5, speed=1.5)

        # 3. Handle word submission (Shift press)
        if shift_pressed and self.selected_path:
            word = "".join([self.grid[y][x] for x, y in self.selected_path]).lower()
            
            if len(word) >= 3 and word in self.WORD_LIST and word not in self.formed_words:
                # Valid word
                word_reward = len(word) - 2
                reward += word_reward
                self.score += len(word)
                self.formed_words.append(word)
                self.word_feedback = {"text": f"+{len(word)}! '{word.upper()}'", "color": self.COLOR_FEEDBACK_GOOD, "alpha": 255}
                self._create_particles((320, 200), self.COLOR_SELECTED_PATH, 50, count=30)
            else:
                # Invalid word
                reward -= 0.5
                self.word_feedback = {"text": "Invalid Word", "color": self.COLOR_FEEDBACK_BAD, "alpha": 255}
            
            self.selected_path = [] # Clear path after submission

        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        if self.word_feedback["alpha"] > 0:
            self.word_feedback["alpha"] = max(0, self.word_feedback["alpha"] - 5)

        # --- Check Termination Conditions ---
        if len(self.formed_words) >= self.WIN_CONDITION_WORDS:
            terminated = True
            self.game_over = True
            reward += 50
            self.win_status = "YOU WIN!"
        elif self.time_remaining <= 0:
            self.time_remaining = 0
            terminated = True
            self.game_over = True
            reward -= 50
            self.win_status = "TIME'S UP!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, 640, self.GRID_TOP_MARGIN - 10))
        
        # Render Grid and Letters
        self._render_grid()
        
        # Render Selection Path and Cursor
        self._render_path_and_cursor()

        # Render Particles
        self._render_particles()

        # Render UI
        self._render_ui()

        # Render Game Over Overlay
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_formed": len(self.formed_words),
            "time_remaining": self.time_remaining,
        }

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_LEFT_MARGIN + x * self.CELL_SIZE,
                    self.GRID_TOP_MARGIN + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)
                
                char = self.grid[y][x]
                color = self.COLOR_LETTER_DEFAULT
                
                # Pulsing effect for selected letters
                if (x, y) in self.selected_path:
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    color = (
                        int(self.COLOR_LETTER_DEFAULT[0] + (self.COLOR_SELECTED_PATH[0] - self.COLOR_LETTER_DEFAULT[0]) * pulse),
                        int(self.COLOR_LETTER_DEFAULT[1] + (self.COLOR_SELECTED_PATH[1] - self.COLOR_LETTER_DEFAULT[1]) * pulse),
                        int(self.COLOR_LETTER_DEFAULT[2] + (self.COLOR_SELECTED_PATH[2] - self.COLOR_LETTER_DEFAULT[2]) * pulse)
                    )

                text_surf = self.font_large.render(char, True, color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

    def _render_path_and_cursor(self):
        # Draw lines connecting selected letters
        if len(self.selected_path) > 1:
            points = [self._cell_to_pixel(pos) for pos in self.selected_path]
            pygame.draw.lines(self.screen, self.COLOR_SELECTED_PATH, False, points, 4)

        # Draw circles on selected letters
        for pos in self.selected_path:
            pixel_pos = self._cell_to_pixel(pos)
            pygame.gfxdraw.aacircle(self.screen, pixel_pos[0], pixel_pos[1], self.CELL_SIZE // 3, self.COLOR_SELECTED_PATH)
            pygame.gfxdraw.filled_circle(self.screen, pixel_pos[0], pixel_pos[1], self.CELL_SIZE // 3, self.COLOR_SELECTED_PATH)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_LEFT_MARGIN + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_TOP_MARGIN + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        alpha = int(128 + 127 * math.sin(self.steps * 0.25))
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Words
        words_text = self.font_medium.render(f"WORDS: {len(self.formed_words)}/{self.WIN_CONDITION_WORDS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(words_text, (20, 50))

        # Timer
        time_color = self.COLOR_UI_TEXT if self.time_remaining > 10 else self.COLOR_TIMER_LOW
        timer_text = self.font_large.render(f"{self.time_remaining:.1f}", True, time_color)
        timer_rect = timer_text.get_rect(right=620, centery=35)
        self.screen.blit(timer_text, timer_rect)

        # Current Word
        current_word_str = "".join([self.grid[y][x] for x, y in self.selected_path])
        word_surf = self.font_medium.render(f"Current: {current_word_str}", True, self.COLOR_SELECTED_PATH)
        word_rect = word_surf.get_rect(centerx=320, top=5)
        self.screen.blit(word_surf, word_rect)
        
        # Word feedback
        if self.word_feedback["alpha"] > 0:
            feedback_surf = self.font_feedback.render(self.word_feedback["text"], True, self.word_feedback["color"])
            feedback_surf.set_alpha(self.word_feedback["alpha"])
            feedback_rect = feedback_surf.get_rect(centerx=320, top=40)
            self.screen.blit(feedback_surf, feedback_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        status_text = self.font_large.render(self.win_status, True, self.COLOR_UI_TEXT)
        status_rect = status_text.get_rect(center=(320, 180))
        self.screen.blit(status_text, status_rect)

        final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        final_score_rect = final_score_text.get_rect(center=(320, 220))
        self.screen.blit(final_score_text, final_score_rect)

    def _generate_grid(self):
        vowels = "AEIOU"
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        # Weighted towards more common letters
        letter_pool = (
            vowels * 15 + "LNRST" * 10 + "DG" * 8 + "BCMP" * 6 +
            "FHVWY" * 4 + "KJX" * 2 + "QZ"
        )
        grid = []
        for _ in range(self.GRID_HEIGHT):
            row = random.choices(letter_pool, k=self.GRID_WIDTH)
            grid.append(row)
        return grid

    def _cell_to_pixel(self, pos):
        x, y = pos
        return (
            self.GRID_LEFT_MARGIN + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_TOP_MARGIN + y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

    def _create_particles(self, pos, color, radius, count=10, speed=2.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = random.uniform(0.5 * speed, 1.5 * speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * vel, math.sin(angle) * vel],
                "radius": random.uniform(radius * 0.5, radius),
                "color": color,
                "life": random.uniform(15, 30) # frames
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["radius"] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0.5]
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Word Grid Game")
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            wait_for_restart = True
            while wait_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_restart = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_restart = False

    env.close()