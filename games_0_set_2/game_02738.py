
# Generated: 2025-08-27T21:17:27.406075
# Source Brief: brief_02738.md
# Brief Index: 2738

        
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
        "Controls: Use arrow keys to select adjacent letters. "
        "Press space to submit a word. Hold shift to clear your current selection."
    )

    game_description = (
        "Connect adjacent letters in the grid to form words. "
        "Score 100 points to win before you run out of your 15 moves."
    )

    auto_advance = False

    # --- Constants ---
    # Visuals
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    GRID_TOP_MARGIN = 60
    GRID_BOTTOM_MARGIN = 80
    
    COLOR_BG = (240, 240, 245) # Light Gray
    COLOR_GRID = (180, 180, 190) # Mid Gray
    COLOR_TEXT = (25, 25, 35) # Dark Slate
    COLOR_SCORE = (60, 60, 70)
    COLOR_SELECTOR = (255, 190, 0, 150) # Semi-transparent Gold
    COLOR_PATH = (60, 220, 120, 120) # Semi-transparent Green
    COLOR_FLASH_VALID = (255, 255, 255) # White
    COLOR_FLASH_INVALID = (255, 80, 80) # Red
    COLOR_WIN = (60, 220, 120, 230)
    COLOR_LOSE = (220, 60, 60, 230)

    # Gameplay
    WIN_SCORE = 100
    STARTING_MOVES = 15
    MAX_STEPS = 1000

    # Word Data (No external files)
    LETTER_FREQUENCIES = "EEEEEEEEEEEEAAAAAAAAAIIIIIIIIIOOOOOOOONNNNNNRRRRRRTTTTTTLLLLSSSSUUUUDDDDGGGBBCCMMPPFFHHVVWWYYKJXQZ"
    WORD_LIST = {
        "ART", "EAR", "RAT", "TAR", "ATE", "EAT", "TEA", "ARE", "ERA", "SEA", "SET",
        "TEN", "NET", "END", "DEN", "AND", "TAN", "ANT", "RAN", "CAR", "ARC", "CAT",
        "ACT", "DOG", "GOD", "LOG", "GOT", "TOP", "POT", "OPT", "RUN", "URN", "SUN",
        "USE", "SUE", "WIN", "OWN", "NOW", "WON", "AIR", "RAIL", "TRAIL", "SAIL",
        "RATE", "TEAR", "CARE", "RACE", "CASE", "SEAT", "EAST", "SAVE", "VASE",
        "STAR", "RATS", "ARTS", "CART", "TRAM", "MART", "PART", "TRAP", "RAPT",
        "SEND", "ENDS", "LEND", "DEAL", "LEAD", "LADE", "DATE", "READ", "DARE",
        "DEAR", "ROAD", "DOOR", "WORD", "LORD", "SOLD", "GOLD", "HOLD", "COLD",
        "BOLD", "BOND", "POND", "FOND", "FIND", "MIND", "KIND", "WIND", "SAND",
        "LAND", "HAND", "BAND", "MORE", "CORE", "SORE", "ROSE", "PORE", "ROPE",
        "HOPE", "HOME", "COME", "SOME", "GAME", "SAME", "MALE", "LAME", "TAME",
        "FAME", "LATE", "TALE", "SALE", "SALT", "LAST", "LOST", "COST", "POST",
        "STOP", "SPOT", "POTS", "TOPS", "PORT", "SORT", "FORT", "FORM", "FROM",
        "WARM", "ARM", "FARM", "HARM", "CHARM", "CHART", "HEART", "EARTH", "START",
        "SMART", "STORE", "STONE", "TONES", "NOTES", "LOAN", "ALON", "ALONG",
        "SONG", "LONG", "SING", "SIGN", "RING", "GRIN", "GRAIN", "RAIN", "TRAIN",
        "BRAIN", "BRAND", "GRAND", "PLANT", "PLAN", "PANT", "PAIN", "PAINT",
        "SAINT", "STAIN", "TRAIN", "QUITE", "QUIET", "QUERY", "QUEST", "REACT",
        "TRACE", "CREATE", "CREATIVE", "ACTIVE", "ACTION", "NATION", "RATIO",
        "MOTION", "POTION", "OPTION", "ORANGE", "GREEN", "BLUE", "BLACK", "WHITE",
        "BROWN", "PLAYER", "AGENT", "SCORE", "MOVES", "GRID", "WORD", "LETTER"
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
        
        self.font_letter = pygame.font.Font(None, 38)
        self.font_ui = pygame.font.Font(None, 28)
        self.font_word = pygame.font.Font(None, 42)
        self.font_game_over = pygame.font.Font(None, 72)

        self.grid_height = self.SCREEN_HEIGHT - self.GRID_TOP_MARGIN - self.GRID_BOTTOM_MARGIN
        self.cell_size = self.grid_height // self.GRID_SIZE
        self.grid_width = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = self.GRID_TOP_MARGIN

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[self._get_random_letter() for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.current_path = []
        
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.steps = 0
        self.game_over = False
        self.win = False

        # Transient state for visual feedback
        self._flash_info = None # Tuple: (path, color)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Every action costs a move
        self.moves_left -= 1
        reward -= 0.01 # Small penalty for using a move

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        action_taken = False
        if space_pressed:
            reward += self._submit_word()
            action_taken = True
        elif shift_pressed:
            if self.current_path:
                self.current_path = []
                self.selector_pos = list(self.initial_selection_pos)
                reward -= 0.5 # Penalty for clearing
                action_taken = True
        elif movement != 0:
            reward += self._move_selector(movement)
            action_taken = True

        if not action_taken:
            reward -= 0.1 # Penalty for no-op

        # Update game over state
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            reward -= 50
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        observation = self._get_observation()
        
        # Clear transient visual state after rendering
        self._flash_info = None

        return (
            observation,
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_random_letter(self):
        return random.choice(self.LETTER_FREQUENCIES)

    def _move_selector(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        new_x = self.selector_pos[0] + dx
        new_y = self.selector_pos[1] + dy

        # Check bounds
        if not (0 <= new_x < self.GRID_SIZE and 0 <= new_y < self.GRID_SIZE):
            return -0.2 # Penalty for hitting wall

        # Start a new path
        if not self.current_path:
            self.current_path.append((new_x, new_y))
            self.initial_selection_pos = (new_x, new_y)
            self.selector_pos = [new_x, new_y]
            return 0.1

        # Check for valid path extension
        last_pos = self.current_path[-1]
        is_adjacent = abs(new_x - last_pos[0]) <= 1 and abs(new_y - last_pos[1]) <= 1
        is_new = (new_x, new_y) not in self.current_path

        if is_adjacent and is_new:
            self.current_path.append((new_x, new_y))
            self.selector_pos = [new_x, new_y]
            return 0.1 # Reward for extending path
        else:
            return -0.1 # Penalty for invalid move

    def _submit_word(self):
        if not self.current_path:
            return -0.2 # Penalty for empty submission

        word = "".join([self.grid[y][x] for x, y in self.current_path])
        
        # Word validation
        is_valid = len(word) >= 3 and word in self.WORD_LIST

        if is_valid:
            # Sound: success_chime.wav
            word_len = len(word)
            points = {3: 1, 4: 2, 5: 4, 6: 6}.get(word_len, 8) # 8 for 7+
            self.score += points
            reward = points

            self._flash_info = (self.current_path, self.COLOR_FLASH_VALID)

            # Remove letters and drop new ones
            for x, y in self.current_path:
                self.grid[y][x] = None
            
            for col in range(self.GRID_SIZE):
                empty_row = self.GRID_SIZE - 1
                for row in range(self.GRID_SIZE - 1, -1, -1):
                    if self.grid[row][col] is not None:
                        self.grid[empty_row][col], self.grid[row][col] = self.grid[row][col], self.grid[empty_row][col]
                        empty_row -= 1
                for row in range(empty_row, -1, -1):
                    self.grid[row][col] = self._get_random_letter()

        else: # Invalid word
            # Sound: error_buzz.wav
            reward = -2
            if len(word) < 3:
                self.score = max(0, self.score - int(self.score * 0.2))
                reward -= 3 # Heavier penalty for short words
            
            self._flash_info = (self.current_path, self.COLOR_FLASH_INVALID)

        # Reset path
        self.current_path = []
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_path_and_selector()
        if self._flash_info:
            self._render_flash(*self._flash_info)
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "current_word": "".join([self.grid[y][x] for x, y in self.current_path]) if self.current_path else ""
        }

    def _render_grid(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GRID)

                letter = self.grid[y][x]
                if letter:
                    text_surf = self.font_letter.render(letter, True, self.COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

    def _render_path_and_selector(self):
        # Draw path
        if self.current_path:
            for i, (x, y) in enumerate(self.current_path):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                # Create a temporary surface for transparency
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                s.fill(self.COLOR_PATH)
                self.screen.blit(s, rect.topleft)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        rect = pygame.Rect(
            self.grid_offset_x + sel_x * self.cell_size,
            self.grid_offset_y + sel_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.gfxdraw.box(s, s.get_rect(), self.COLOR_SELECTOR)
        self.screen.blit(s, rect.topleft)

    def _render_flash(self, path, color):
        for x, y in path:
            rect = pygame.Rect(
                self.grid_offset_x + x * self.cell_size,
                self.grid_offset_y + y * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Re-render letter on top of flash
            letter = self.grid[y][x]
            if letter:
                text_surf = self.font_letter.render(letter, True, self.COLOR_TEXT if color != self.COLOR_FLASH_VALID else self.COLOR_BG)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (20, 20))

        # Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_SCORE)
        moves_rect = moves_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(moves_surf, moves_rect)

        # Current Word
        word_str = "".join([self.grid[y][x] for x, y in self.current_path]) if self.current_path else ""
        word_surf = self.font_word.render(word_str, True, self.COLOR_TEXT)
        word_rect = word_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40))
        self.screen.blit(word_surf, word_rect)
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        
        overlay.fill(color)
        
        text_surf = self.font_game_over.render(text, True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("LexiGrid")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        # Human input mapping
        move_action = 0 # None
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: move_action = 1
                elif event.key == pygame.K_DOWN: move_action = 2
                elif event.key == pygame.K_LEFT: move_action = 3
                elif event.key == pygame.K_RIGHT: move_action = 4
                elif event.key == pygame.K_SPACE: space_action = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 1
                elif event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action was taken
        if move_action or space_action or shift_action:
            action = [move_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if env.game_over:
            # Pause on game over, wait for R to reset or Esc to quit
            pass

    env.close()