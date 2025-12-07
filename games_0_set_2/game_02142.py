
# Generated: 2025-08-27T19:24:03.707490
# Source Brief: brief_02142.md
# Brief Index: 2142

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to select a letter. Shift to submit a word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find and form words by connecting adjacent letters on the grid. Find 5 words before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_TILE_DEFAULT = (60, 90, 120)
    COLOR_TILE_USED = (30, 45, 60)
    COLOR_TEXT_DEFAULT = (220, 220, 230)
    COLOR_TEXT_USED = (80, 80, 90)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (0, 180, 120)
    COLOR_SELECTION_LINE = (0, 220, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_SUCCESS = (100, 255, 150)
    COLOR_FAIL = (255, 100, 100)

    # Grid
    GRID_COLS, GRID_ROWS = 10, 7
    TILE_SIZE = 48
    TILE_MARGIN = 4
    GRID_WIDTH = GRID_COLS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_HEIGHT = GRID_ROWS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_TOP = 80
    GRID_LEFT = (640 - GRID_WIDTH) // 2

    # Game rules
    WIN_WORD_COUNT = 5
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Word list (self-contained to avoid file I/O)
    WORD_LIST = {
        "cat", "dog", "sun", "run", "big", "red", "bed", "egg", "fun", "get", "gym", "hat", "hen", "jam", "jet", "key",
        "leg", "map", "net", "pen", "pig", "rat", "sad", "sit", "ten", "top", "toy", "van", "vet", "web", "zoo", "able",
        "acid", "aged", "also", "area", "army", "away", "baby", "back", "ball", "band", "bank", "base", "bean", "bear",
        "beat", "bell", "best", "bird", "blow", "blue", "boat", "body", "bomb", "bond", "bone", "book", "boom", "born",
        "boss", "both", "bowl", "burn", "bush", "busy", "call", "calm", "came", "camp", "card", "care", "case", "cash",
        "cast", "cell", "chat", "chip", "city", "club", "coal", "coat", "code", "cold", "come", "cook", "cool", "cope",
        "core", "cost", "crew", "crop", "dark", "data", "date", "dawn", "dead", "deal", "dean", "dear", "debt", "deck",
        "deep", "deer", "desk", "dirt", "door", "down", "draw", "dream", "drop", "duck", "duke", "duty", "each", "earn",
        "east", "easy", "edge", "else", "even", "ever", "evil", "exit", "face", "fact", "fail", "fair", "fall", "farm",
        "fast", "fate", "fear", "feed", "feel", "file", "fill", "film", "find", "fine", "fire", "firm", "fish", "five",
        "flat", "flow", "food", "foot", "form", "four", "free", "from", "fuel", "full", "fund", "gain", "game", "gate",
        "gaze", "gear", "gene", "gift", "girl", "give", "glad", "goal", "goes", "gold", "golf", "good", "grab", "gray",
        "great", "green", "grey", "grid", "grow", "gulf", "hair", "half", "hall", "hand", "hang", "hard", "harm", "hate",
        "have", "head", "hear", "heat", "held", "hell", "help", "here", "hero", "high", "hill", "hire", "hold", "hole",
        "holy", "home", "hope", "host", "hour", "huge", "hunt", "hurt", "idea", "inch", "into", "iron", "item", "jail",
        "join", "jump", "jury", "just", "keep", "kill", "king", "knee", "knew", "know", "lack", "lady", "laid", "lake",
        "land", "lane", "last", "late", "lead", "left", "less", "life", "lift", "like", "line", "link", "list", "live",
        "load", "loan", "lock", "logo", "long", "look", "lord", "lose", "loss", "lost", "love", "luck", "made", "mail",
        "main", "make", "male", "many", "mark", "mass", "meal", "mean", "meat", "meet", "menu", "mere", "mild", "mile",
        "milk", "mind", "mine", "miss", "mode", "mood", "moon", "more", "most", "move", "much", "must", "name", "navy",
        "near", "neck", "need", "next", "nice", "nine", "none", "nose", "note", "okay", "once", "only", "onto", "open",
        "oral", "over", "pace", "pack", "page", "paid", "pain", "pair", "palm", "park", "part", "pass", "past", "path",
        "peak", "pick", "pink", "pipe", "plan", "play", "plot", "plug", "plus", "poem", "poet", "pole", "poll", "pool",
        "poor", "port", "post", "pull", "pure", "push", "race", "rail", "rain", "rank", "rare", "rate", "read", "real",
        "rely", "rest", "rice", "rich", "ride", "ring", "rise", "risk", "road", "rock", "role", "roll", "roof", "room",
        "root", "rose", "rule", "rush", "safe", "said", "sake", "sale", "salt", "same", "sand", "save", "seat", "seed",
        "seek", "seem", "sell", "send", "sense", "sent", "sept", "ship", "shop", "shot", "show", "sick", "side", "sign",
        "sing", "site", "size", "skin", "slip", "slow", "snow", "soft", "soil", "sole", "some", "song", "soon", "sort",
        "soul", "spot", "star", "stay", "step", "stop", "such", "suit", "sure", "take", "tale", "talk", "tall", "tank",
        "tape", "task", "team", "tear", "tech", "tell", "tend", "term", "test", "text", "than", "that", "thee", "then",
        "they", "thin", "this", "thus", "tide", "tile", "time", "tiny", "told", "toll", "tone", "tool", "tour", "town",
        "tree", "trip", "true", "tube", "turn", "twin", "type", "unit", "upon", "used", "user", "vast", "very", "veto",
        "view", "vote", "wage", "wait", "wake", "walk", "wall", "want", "warm", "warn", "wash", "wave", "weak", "wear",
        "week", "well", "went", "were", "west", "what", "when", "whom", "wide", "wife", "wild", "will", "wind", "wine",
        "wing", "wire", "wise", "wish", "with", "wood", "word", "work", "yard", "yeah", "year", "your", "zero", "zone",
        "agent", "board", "boost", "chair", "chaos", "chest", "class", "clear", "clone", "cloud", "coast", "craft",
        "cream", "crime", "cycle", "dance", "death", "delta", "depth", "drift", "drive", "earth", "enemy", "error",
        "event", "extra", "faith", "field", "final", "fight", "floor", "focus", "force", "frame", "front", "fruit",
        "giant", "glass", "grade", "grand", "grass", "grave", "grief", "guard", "guest", "guide", "heart", "heavy",
        "horse", "hotel", "house", "human", "image", "index", "input", "issue", "judge", "knife", "large", "laser",
        "layer", "level", "light", "limit", "local", "logic", "magic", "major", "march", "match", "maybe", "metal",
        "model", "money", "month", "motor", "mouse", "mouth", "music", "nerve", "never", "night", "noise", "north",
        "novel", "nurse", "ocean", "offer", "order", "other", "owner", "panel", "paper", "party", "peace", "phase",
        "phone", "photo", "piece", "pilot", "pitch", "place", "plane", "plant", "plate", "point", "pound", "power",
        "press", "price", "pride", "print", "prize", "proof", "proud", "pulse", "queen", "quest", "quiet", "quite",
        "radio", "raise", "range", "rapid", "ratio", "reach", "react", "ready", "reply", "right", "river", "robot",
        "round", "route", "royal", "scale", "scene", "scope", "score", "sense", "serve", "shade", "shape", "share",
        "sharp", "sheet", "shelf", "shell", "shift", "shine", "shirt", "shock", "shoot", "short", "sight", "sigma",
        "skill", "sleep", "slice", "small", "smart", "smile", "smoke", "solid", "solve", "sorry", "sound", "south",
        "space", "speak", "speed", "spend", "spite", "split", "sport", "squad", "stack", "staff", "stage", "stand",
        "stare", "start", "state", "steam", "steel", "stick", "still", "stock", "stone", "store", "storm", "story",
        "strip", "study", "stuff", "style", "sugar", "super", "sweet", "table", "taste", "teach", "thank", "theme",
        "there", "these", "thing", "think", "three", "throw", "tiger", "timer", "title", "total", "touch", "tower",
        "trace", "track", "trade", "train", "trash", "treat", "trial", "tribe", "trick", "troop", "truck", "truly",
        "trust", "truth", "twice", "ultra", "uncle", "under", "union", "unite", "until", "upper", "urban", "usual",
        "value", "vapor", "video", "virus", "visit", "vital", "voice", "waste", "watch", "water", "wheel", "where",
        "which", "while", "white", "whole", "woman", "world", "worry", "worth", "wound", "write", "wrong", "yield",
        "young", "youth",
    }
    LETTER_FREQUENCIES = "eeeeeeeeeeeeaaaaaaaaaiiiiiiiiioooooooonnnnnnrrrrrrttttttllllssssuuuuddddgggbbccmmppffhhvvwwyykjxqz"

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
        
        self.font_tile = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_status = pygame.font.SysFont("Arial", 20)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.found_words = []
        self.particles = []
        self.status_message = ("", (0,0,0), 0) # text, color, lifetime

        self.last_space_held = False
        self.last_shift_held = False
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _generate_grid(self):
        grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        # Attempt to place a few guaranteed words
        words_to_place = random.sample(sorted([w for w in self.WORD_LIST if 4 <= len(w) <= 6]), k=5)
        
        for word in words_to_place:
            for _ in range(20): # 20 placement attempts per word
                path = self._find_path_for_word(grid, word)
                if path:
                    for i, (r, c) in enumerate(path):
                        grid[r][c] = word[i].upper()
                    break

        # Fill remaining empty cells
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if grid[r][c] == '':
                    grid[r][c] = random.choice(self.LETTER_FREQUENCIES).upper()
        
        return grid

    def _find_path_for_word(self, grid, word):
        rows, cols = len(grid), len(grid[0])
        for _ in range(50): # 50 start position attempts
            path = []
            r, c = self.np_random.integers(0, rows), self.np_random.integers(0, cols)
            
            if grid[r][c] != '':
                continue

            path.append((r, c))
            
            for i in range(1, len(word)):
                neighbors = self._get_valid_neighbors(grid, path[-1], path)
                if not neighbors:
                    break
                path.append(random.choice(neighbors))
            
            if len(path) == len(word):
                return path
        return None

    def _get_valid_neighbors(self, grid, pos, excluded):
        r, c = pos
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and grid[nr][nc] == '' and (nr, nc) not in excluded:
                    neighbors.append((nr, nc))
        return neighbors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self.grid = self._generate_grid()
        self.used_tiles = set()
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_path = []
        self.found_words = []
        self.particles = []

        self.last_space_held = False
        self.last_shift_held = False

        self.status_message = ("Find 5 words!", self.COLOR_UI_TEXT, 120)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        if movement == 2: self.cursor_pos[0] += 1  # Down
        if movement == 3: self.cursor_pos[1] -= 1  # Left
        if movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_ROWS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_COLS - 1)

        # Space (Select letter)
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            r, c = self.cursor_pos
            pos = (r, c)
            
            if pos in self.used_tiles:
                # Cannot select used tile
                pass # sfx: invalid_buzz
            elif pos in self.selected_path:
                # Deselect back to this tile
                idx = self.selected_path.index(pos)
                self.selected_path = self.selected_path[:idx + 1]
            else:
                # Add new tile if adjacent
                if not self.selected_path or self._is_adjacent(pos, self.selected_path[-1]):
                    self.selected_path.append(pos)
                    # sfx: select_click
                else:
                    # Invalid selection (not adjacent)
                    reward -= 0.1
                    self.status_message = ("Not adjacent!", self.COLOR_FAIL, 60)
                    # sfx: error_sound

        # Shift (Submit word)
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed and self.selected_path:
            self.moves_left -= 1
            word = "".join([self.grid[r][c] for r, c in self.selected_path]).lower()

            if len(word) > 2 and word in self.WORD_LIST and word not in self.found_words:
                # Valid word
                self.found_words.append(word)
                word_score = len(word) * 10
                self.score += word_score
                reward += 1.0
                
                if len(word) > 4:
                    reward += 5.0
                    self.score += 25 # Bonus points
                    self.status_message = (f"'{word.upper()}' +{word_score} (+25 Bonus!)", self.COLOR_SUCCESS, 90)
                else:
                    self.status_message = (f"'{word.upper()}' +{word_score}", self.COLOR_SUCCESS, 90)

                for r, c in self.selected_path:
                    self.used_tiles.add((r, c))
                    self._create_particles((r,c))
                # sfx: word_success
            else:
                # Invalid word
                reward -= 0.5
                if word in self.found_words:
                    self.status_message = ("Already found!", self.COLOR_FAIL, 60)
                else:
                    self.status_message = ("Not a word!", self.COLOR_FAIL, 60)
                # sfx: word_fail
            
            self.selected_path = []
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and len(self.found_words) >= self.WIN_WORD_COUNT:
            reward += 50.0 # Big reward for winning

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.game_over: return True
        
        win = len(self.found_words) >= self.WIN_WORD_COUNT
        loss_moves = self.moves_left <= 0
        loss_steps = self.steps >= self.MAX_STEPS
        
        if win:
            self.game_over = True
            self.status_message = ("YOU WIN!", self.COLOR_SUCCESS, 300)
        elif loss_moves or loss_steps:
            self.game_over = True
            self.status_message = ("GAME OVER", self.COLOR_FAIL, 300)
            
        return self.game_over

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "words_found": len(self.found_words)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                pygame.draw.circle(self.screen, p['color'] + (alpha,), [int(x) for x in p['pos']], int(p['size']))

        # Draw grid tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                is_used = (r, c) in self.used_tiles
                is_selected = (r, c) in self.selected_path
                
                tile_color = self.COLOR_TILE_DEFAULT
                text_color = self.COLOR_TEXT_DEFAULT
                if is_used:
                    tile_color = self.COLOR_TILE_USED
                    text_color = self.COLOR_TEXT_USED
                elif is_selected:
                    tile_color = self.COLOR_SELECTION
                
                rect = self._get_tile_rect(r, c)
                pygame.draw.rect(self.screen, tile_color, rect, border_radius=5)
                
                letter = self.grid[r][c]
                text_surf = self.font_tile.render(letter, True, text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

        # Draw selection line
        if len(self.selected_path) > 1:
            points = [self._get_tile_rect(r, c).center for r, c in self.selected_path]
            pygame.draw.lines(self.screen, self.COLOR_SELECTION_LINE, False, points, 5)

        # Draw cursor
        cursor_rect = self._get_tile_rect(*self.cursor_pos)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Moves Left
        moves_text = f"Moves Left: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        moves_rect = moves_surf.get_rect(topright=(640 - 15, 10))
        self.screen.blit(moves_surf, moves_rect)

        # Current Word
        current_word = "".join([self.grid[r][c] for r, c in self.selected_path])
        word_text = f"Current: {current_word}"
        word_surf = self.font_ui.render(word_text, True, self.COLOR_UI_TEXT)
        word_rect = word_surf.get_rect(midtop=(320, 10))
        self.screen.blit(word_surf, word_rect)
        
        # Found Words
        found_text = f"Found: {len(self.found_words)}/{self.WIN_WORD_COUNT}"
        found_surf = self.font_ui.render(found_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(found_surf, (15, 40))

        # Status Message
        if self.status_message[2] > 0:
            msg, color, life = self.status_message
            alpha = min(255, life * 5)
            status_surf = self.font_status.render(msg, True, color)
            status_surf.set_alpha(alpha)
            status_rect = status_surf.get_rect(center=(320, 380))
            self.screen.blit(status_surf, status_rect)
            self.status_message = (msg, color, life - 1)
        
        # Game Over Screen
        if self.game_over and self.status_message[2] <= 0:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            
            msg, color, _ = ("YOU WIN!", self.COLOR_SUCCESS, 0) if len(self.found_words) >= self.WIN_WORD_COUNT else ("GAME OVER", self.COLOR_FAIL, 0)
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(320, 200))

            self.screen.blit(overlay, (0,0))
            self.screen.blit(text_surf, text_rect)

    def _get_tile_rect(self, r, c):
        x = self.GRID_LEFT + c * (self.TILE_SIZE + self.TILE_MARGIN)
        y = self.GRID_TOP + r * (self.TILE_SIZE + self.TILE_MARGIN)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)

    def _create_particles(self, pos_rc):
        center = self._get_tile_rect(*pos_rc).center
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(center),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'max_life': 40,
                'size': random.uniform(2, 5),
                'color': self.COLOR_SUCCESS
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Set up a window to display the environment
    pygame.display.set_caption("Word Grid Game")
    screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls for Human Player ---
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Words Found: {info['words_found']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # The environment is not auto-advancing, so we control the frame rate here
        env.clock.tick(30)
        
    env.close()