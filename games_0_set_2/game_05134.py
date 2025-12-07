import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place letter. Shift to cycle through your hand."
    )

    game_description = (
        "Place letters on the grid to form words horizontally and vertically. Create 5 words before the timer runs out to win!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    CELL_SIZE = 32
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_TOP_LEFT = ((SCREEN_WIDTH - GRID_WIDTH) // 2, (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20)

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (50, 65, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_VALID_WORD = (0, 255, 120)
    COLOR_PLACED_LETTER = (255, 255, 255)
    COLOR_GAMEOVER_WIN = (100, 255, 150)
    COLOR_GAMEOVER_LOSS = (255, 100, 100)

    # --- Game Settings ---
    GAME_DURATION_STEPS = 600
    WIN_CONDITION_WORDS = 5
    HAND_SIZE = 5
    MAX_EPISODE_STEPS = 1000

    # --- Word & Letter Data (No external files) ---
    WORD_LIST = {
        "cat", "dog", "sun", "run", "big", "red", "bed", "ace", "act", "add", "age", "ago", "aid", "aim", "air", "ale", "all", "and", "any", "ape", "apt", "arc", "are", "ark", "arm", "art", "ash", "ask", "ate", "awe",
        "bad", "bag", "ban", "bat", "bee", "beg", "bet", "bid", "bin", "bit", "boa", "bob", "bog", "boo", "bow", "box", "boy", "bud", "bug", "bum", "bun", "bus", "but", "buy", "bye",
        "cab", "cad", "cam", "can", "cap", "car", "cod", "cog", "con", "coo", "cop", "cot", "cow", "coy", "cry", "cub", "cud", "cue", "cup", "cur", "cut",
        "dad", "dam", "day", "den", "dew", "did", "die", "dig", "dim", "din", "dip", "doe", "dot", "dry", "dub", "dud", "due", "dug", "dun", "duo",
        "ear", "eat", "eel", "egg", "ego", "elf", "elk", "elm", "end", "eon", "era", "erg", "eve", "eye",
        "fade", "fall", "farm", "fast", "fate", "find", "fine", "fire", "firm", "fish", "five", "flag", "flat", "flow", "food", "foot", "form", "four", "free", "from", "fuel", "full",
        "game", "gate", "give", "glad", "glow", "goal", "goes", "gold", "good", "grid", "grow", "gym",
        "hail", "hair", "half", "hand", "hang", "hard", "have", "head", "hear", "heat", "held", "help", "here", "hide", "high", "hike", "hold", "hole", "home", "hope", "hour", "huge", "hunt", "hurt",
        "idea", "idle", "into", "iron", "item",
        "join", "joke", "jump", "just",
        "keep", "key", "kick", "kill", "kind", "king", "kiss", "knee", "knew", "know",
        "lack", "lady", "laid", "lake", "land", "last", "late", "lava", "lead", "leaf", "left", "lend", "less", "letter", "life", "like", "line", "list", "live", "load", "long", "look", "lose", "lost", "love", "luck",
        "made", "mail", "main", "make", "many", "map", "mark", "mass", "maze", "mean", "meet", "menu", "mild", "mind", "mine", "miss", "mode", "moon", "more", "most", "move", "much",
        "nail", "name", "near", "neck", "need", "next", "nice", "nine", "none", "nose", "note", "noun",
        "once", "one", "only", "onto", "open", "over",
        "pack", "page", "pain", "pair", "park", "part", "pass", "past", "path", "play", "plot", "plus", "poem", "pole", "pond", "pool", "poor", "post", "pull", "pure", "push",
        "race", "rain", "rank", "rate", "read", "real", "rely", "rest", "rich", "ride", "ring", "rise", "risk", "road", "rock", "role", "roll", "roof", "room", "root", "rope", "rose", "ruby", "rule",
        "safe", "said", "sail", "sale", "salt", "same", "sand", "save", "scan", "score", "seat", "seed", "seek", "seem", "see", "self", "sell", "send", "sent", "set", "seven", "ship", "shot", "show", "side", "sign", "sing", "size", "skill", "skin", "slow", "snow", "soft", "soil", "some", "song", "soon", "sort", "space", "spin", "spot", "star", "stat", "step", "stop", "such", "suit", "sure",
        "take", "talk", "tall", "task", "team", "tell", "tend", "term", "test", "text", "than", "that", "the", "them", "then", "they", "thin", "this", "time", "tiny", "told", "tool", "top", "total", "town", "track", "tree", "true", "turn", "two", "type",
        "unit", "until", "use", "user",
        "vast", "very", "view", "void", "vote",
        "wait", "walk", "wall", "want", "warm", "was", "wave", "way", "weak", "wear", "week", "well", "went", "were", "what", "when", "who", "why", "wide", "wife", "wild", "will", "win", "wind", "wine", "wing", "wise", "wish", "with", "word", "work", "world",
        "year", "yes", "yet", "you", "your",
        "zero", "zone",
        "agent", "action", "expert", "final", "puzzle", "python", "quality", "reward", "state", "visual",
    }

    LETTER_FREQUENCIES = {
        'A': 8.17, 'B': 1.49, 'C': 2.78, 'D': 4.25, 'E': 12.70,
        'F': 2.23, 'G': 2.02, 'H': 6.09, 'I': 6.97, 'J': 0.15,
        'K': 0.77, 'L': 4.03, 'M': 2.41, 'N': 6.75, 'O': 7.51,
        'P': 1.93, 'Q': 0.10, 'R': 5.99, 'S': 6.33, 'T': 9.06,
        'U': 2.76, 'V': 0.98, 'W': 2.36, 'X': 0.15, 'Y': 1.97, 'Z': 0.07
    }

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
        
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_grid = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Impact", 60)
        self.font_popup = pygame.font.SysFont("Consolas", 16, bold=True)

        self.grid = []
        self.cursor_pos = [0, 0]
        self.hand = []
        self.selected_letter_idx = 0
        self.time_left = 0
        self.words_found_count = 0
        self.found_words = set()
        self.visual_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        letters, weights = zip(*self.LETTER_FREQUENCIES.items())
        self.hand = self.rng.choices(population=letters, weights=weights, k=self.HAND_SIZE)
        
        self.selected_letter_idx = 0
        self.time_left = self.GAME_DURATION_STEPS
        self.words_found_count = 0
        self.found_words = set()
        
        self.visual_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        # Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        
        # Cycle letter (Shift press on rising edge)
        if shift_held and not self.prev_shift_held and len(self.hand) > 0:
            self.selected_letter_idx = (self.selected_letter_idx + 1) % len(self.hand)

        # Place letter (Space press on rising edge)
        if space_held and not self.prev_space_held:
            row, col = self.cursor_pos[1], self.cursor_pos[0]
            if self.grid[row][col] == '' and len(self.hand) > 0:
                letter_to_place = self.hand.pop(self.selected_letter_idx)
                self.grid[row][col] = letter_to_place
                
                # Check for new words
                newly_formed_words, word_positions = self._check_for_words((row, col))
                
                if newly_formed_words:
                    word_reward = 0
                    for word in newly_formed_words:
                        word_len = len(word)
                        word_reward += word_len # +1 per letter
                        if word_len >= 4:
                            word_reward += 10 # Bonus for long words
                    
                    self.score += word_reward
                    reward += word_reward
                    self.words_found_count += len(newly_formed_words)
                    
                    # Add visual effects for found words
                    self._add_visual_effect('word_highlight', positions=word_positions, duration=45)
                    self._add_visual_effect('score_popup', pos=(col, row), text=f"+{word_reward}", duration=30, color=self.COLOR_VALID_WORD)
                else:
                    reward -= 0.2 # Penalty for non-word placement

                if len(self.hand) > 0:
                    self.selected_letter_idx = self.selected_letter_idx % len(self.hand)
                else:
                    self.selected_letter_idx = 0
            else:
                # Invalid placement (cell occupied or hand empty)
                self._add_visual_effect('invalid_flash', pos=self.cursor_pos, duration=15)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        self.time_left -= 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.words_found_count >= self.WIN_CONDITION_WORDS:
            reward += 50
            terminated = True
            self.termination_reason = "YOU WIN!"
            self.game_over = True
        elif self.time_left <= 0:
            reward -= 50
            terminated = True
            self.termination_reason = "TIME'S UP!"
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is that truncated episodes are also terminated
            self.termination_reason = "STEP LIMIT REACHED"
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_for_words(self, pos):
        r, c = pos
        newly_formed_words = []
        all_word_positions = set()

        # Horizontal check
        start_c = c
        while start_c > 0 and self.grid[r][start_c - 1] != '':
            start_c -= 1
        end_c = c
        while end_c < self.GRID_SIZE - 1 and self.grid[r][end_c + 1] != '':
            end_c += 1
        
        if start_c != end_c:
            word_str = "".join(self.grid[r][i] for i in range(start_c, end_c + 1))
            if word_str.lower() in self.WORD_LIST and word_str not in self.found_words:
                newly_formed_words.append(word_str)
                self.found_words.add(word_str)
                for i in range(start_c, end_c + 1):
                    all_word_positions.add((r, i))

        # Vertical check
        start_r = r
        while start_r > 0 and self.grid[start_r - 1][c] != '':
            start_r -= 1
        end_r = r
        while end_r < self.GRID_SIZE - 1 and self.grid[end_r + 1][c] != '':
            end_r += 1

        if start_r != end_r:
            word_str = "".join(self.grid[i][c] for i in range(start_r, end_r + 1))
            if word_str.lower() in self.WORD_LIST and word_str not in self.found_words:
                newly_formed_words.append(word_str)
                self.found_words.add(word_str)
                for i in range(start_r, end_r + 1):
                    all_word_positions.add((i, c))
                    
        return newly_formed_words, list(all_word_positions)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_effects()
        self._render_grid_and_letters()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_letters(self):
        gx, gy = self.GRID_TOP_LEFT
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (gx + i * self.CELL_SIZE, gy), (gx + i * self.CELL_SIZE, gy + self.GRID_HEIGHT), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (gx, gy + i * self.CELL_SIZE), (gx + self.GRID_WIDTH, gy + i * self.CELL_SIZE), 1)

        # Draw letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter = self.grid[r][c]
                if letter:
                    letter_surf = self.font_grid.render(letter, True, self.COLOR_PLACED_LETTER)
                    letter_rect = letter_surf.get_rect(center=(gx + c * self.CELL_SIZE + self.CELL_SIZE // 2, gy + r * self.CELL_SIZE + self.CELL_SIZE // 2))
                    self.screen.blit(letter_surf, letter_rect)

    def _render_cursor(self):
        if self.game_over: return
        gx, gy = self.GRID_TOP_LEFT
        c, r = self.cursor_pos
        
        # Pulsing alpha
        alpha = 100 + int(math.sin(pygame.time.get_ticks() * 0.005) * 40)
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        cursor_surface.fill((*self.COLOR_CURSOR, alpha))
        self.screen.blit(cursor_surface, (gx + c * self.CELL_SIZE, gy + r * self.CELL_SIZE))

        # Show selected letter in cursor
        if self.hand and self.selected_letter_idx < len(self.hand):
            letter = self.hand[self.selected_letter_idx]
            letter_surf = self.font_grid.render(letter, True, self.COLOR_PLACED_LETTER)
            letter_rect = letter_surf.get_rect(center=(gx + c * self.CELL_SIZE + self.CELL_SIZE // 2, gy + r * self.CELL_SIZE + self.CELL_SIZE // 2))
            self.screen.blit(letter_surf, letter_rect)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))
        
        # Time
        time_sec = max(0, self.time_left // 10)
        time_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_GAMEOVER_LOSS
        time_surf = self.font_ui.render(f"TIME: {time_sec}", True, time_color)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(time_surf, time_rect)
        
        # Words
        words_surf = self.font_ui.render(f"WORDS: {self.words_found_count} / {self.WIN_CONDITION_WORDS}", True, self.COLOR_TEXT)
        words_rect = words_surf.get_rect(midtop=(self.SCREEN_WIDTH // 2, 20))
        self.screen.blit(words_surf, words_rect)
        
        # Hand
        hand_text = "HAND: "
        hand_surf = self.font_ui.render(hand_text, True, self.COLOR_TEXT)
        self.screen.blit(hand_surf, (20, self.SCREEN_HEIGHT - 30))
        
        start_x = 20 + hand_surf.get_width()
        for i, letter in enumerate(self.hand):
            is_selected = (i == self.selected_letter_idx)
            color = self.COLOR_CURSOR if is_selected else self.COLOR_TEXT
            letter_surf = self.font_ui.render(letter, True, color)
            
            if is_selected:
                underline_rect = pygame.Rect(start_x, self.SCREEN_HEIGHT - 12, letter_surf.get_width(), 2)
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, underline_rect)
            
            self.screen.blit(letter_surf, (start_x, self.SCREEN_HEIGHT - 30))
            start_x += letter_surf.get_width() + 10

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        color = self.COLOR_GAMEOVER_WIN if self.words_found_count >= self.WIN_CONDITION_WORDS else self.COLOR_GAMEOVER_LOSS
        text_surf = self.font_gameover.render(self.termination_reason, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _add_visual_effect(self, effect_type, **kwargs):
        effect = {'type': effect_type, **kwargs}
        self.visual_effects.append(effect)

    def _update_and_render_effects(self):
        gx, gy = self.GRID_TOP_LEFT
        
        for effect in self.visual_effects[:]:
            effect['duration'] -= 1
            if effect['duration'] <= 0:
                self.visual_effects.remove(effect)
                continue

            if effect['type'] == 'word_highlight':
                alpha = int(100 * (effect['duration'] / 45))
                highlight_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                highlight_surf.fill((*self.COLOR_VALID_WORD, alpha))
                for r, c in effect['positions']:
                    self.screen.blit(highlight_surf, (gx + c * self.CELL_SIZE, gy + r * self.CELL_SIZE))

            elif effect['type'] == 'invalid_flash':
                alpha = int(200 * (effect['duration'] / 15))
                flash_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surf.fill((*self.COLOR_GAMEOVER_LOSS, alpha))
                c, r = effect['pos']
                self.screen.blit(flash_surf, (gx + c * self.CELL_SIZE, gy + r * self.CELL_SIZE))
            
            elif effect['type'] == 'score_popup':
                progress = 1 - (effect['duration'] / 30)
                c, r = effect['pos']
                start_y = gy + r * self.CELL_SIZE + self.CELL_SIZE // 2
                current_y = start_y - int(20 * progress)
                alpha = 255 - int(255 * progress)
                
                popup_surf = self.font_popup.render(effect['text'], True, (*effect['color'], alpha))
                popup_rect = popup_surf.get_rect(center=(gx + c * self.CELL_SIZE + self.CELL_SIZE // 2, current_y))
                self.screen.blit(popup_surf, popup_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "words_found": self.words_found_count,
            "hand_size": len(self.hand),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # To use this, you might need to unset the dummy video driver
    # and install pygame.
    # For example:
    # unset SDL_VIDEODRIVER
    # pip install pygame
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The main game loop requires a display
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Word Grid")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
            space_held = 0
            shift_held = 0
            
            # For auto_advance=False, we only step on an event
            action_taken = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    action_taken = True # Register key presses for stepping
                    if event.key == pygame.K_r:
                        obs, info = env.reset()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            # Only step if a key is pressed, for turn-based feel
            if any(keys) or action_taken:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")
                    # Keep rendering the final state, but don't step
                    # Wait for 'r' to reset or quit
                    while True:
                        final_event = pygame.event.wait()
                        if final_event.type == pygame.QUIT:
                            running = False
                            break
                        if final_event.type == pygame.KEYDOWN and final_event.key == pygame.K_r:
                            obs, info = env.reset()
                            break
                    if not running:
                        break


            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Limit frame rate for human play
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because the environment is running in headless mode.")
        print("To run interactively, you might need to unset SDL_VIDEODRIVER.")
        
    env.close()