
# Generated: 2025-08-27T13:10:41.390060
# Source Brief: brief_00286.md
# Brief Index: 286

        
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
    user_guide = "Controls: Arrows to move selected tile. Space to select/cycle tiles. Shift to submit sentence."

    # Must be a short, user-facing description of the game:
    game_description = "Unscramble words to form grammatically correct sentences against the clock."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    SENTENCES_TO_WIN = 3

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 45, 60)
    COLOR_TILE = (70, 130, 180)  # Steel Blue
    COLOR_TILE_TEXT = (230, 240, 250)
    COLOR_TILE_SELECTED = (255, 165, 0)  # Orange
    COLOR_TILE_PLACED = (60, 179, 113)  # Medium Sea Green
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (220, 50, 50)
    COLOR_FEEDBACK_CORRECT = (50, 205, 50, 150)
    COLOR_FEEDBACK_INCORRECT = (255, 69, 0, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_tile = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Sentence data
        self.sentence_pool = [
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            "GYMNASIUM IS A GREAT TOOL FOR RL RESEARCH",
            "PYTHON IS A VERSATILE PROGRAMMING LANGUAGE",
            "VISUAL QUALITY ENHANCES THE GAMEPLAY EXPERIENCE",
            "REINFORCEMENT LEARNING CAN SOLVE COMPLEX TASKS",
            "AN AGENT LEARNS BY INTERACTING WITH AN ENVIRONMENT",
            "MAKE THE GAME VISUALLY APPEALING AND FUN",
        ]
        
        # Game layout
        self.sentence_area = pygame.Rect(20, 300, self.SCREEN_WIDTH - 40, 80)
        self.holding_area = pygame.Rect(20, 80, self.SCREEN_WIDTH - 40, 200)

        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.sentences_solved = None
        self.current_sentence_text = None
        self.word_tiles = None
        self.selected_tile_index = None
        self.last_space_held = None
        self.last_shift_held = None
        self.feedback_flash = None
        self.rng = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None:
            self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.sentences_solved = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_flash = None
        
        self._setup_new_puzzle()
        
        return self._get_observation(), self._get_info()

    def _setup_new_puzzle(self):
        self.current_sentence_text = self.rng.choice(self.sentence_pool)
        words = self.current_sentence_text.split()
        
        self.word_tiles = []
        self.selected_tile_index = None
        
        for i, word_text in enumerate(words):
            text_surf = self.font_tile.render(word_text, True, self.COLOR_TILE_TEXT)
            tile_width = text_surf.get_width() + 20
            tile_height = text_surf.get_height() + 10
            
            # Distribute tiles randomly in the holding area
            pos_x = self.rng.integers(
                self.holding_area.left, self.holding_area.right - tile_width
            )
            pos_y = self.rng.integers(
                self.holding_area.top, self.holding_area.bottom - tile_height
            )

            tile = {
                "text": word_text,
                "pos": pygame.Vector2(pos_x, pos_y),
                "rect": pygame.Rect(pos_x, pos_y, tile_width, tile_height),
                "surf": text_surf,
                "is_placed": False,
            }
            self.word_tiles.append(tile)

    def step(self, action):
        reward = 0
        self.game_over = False

        # Unpack factorized action
        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and not self.last_space_held
        shift_pressed = shift_action == 1 and not self.last_shift_held
        self.last_space_held = space_action == 1
        self.last_shift_held = shift_action == 1
        
        # Handle tile selection
        if space_pressed:
            # sound: tile_select.wav
            if self.selected_tile_index is None:
                self.selected_tile_index = 0
            else:
                self.selected_tile_index = (self.selected_tile_index + 1) % len(self.word_tiles)
        
        # Handle tile movement
        if self.selected_tile_index is not None:
            tile = self.word_tiles[self.selected_tile_index]
            move_speed = 8
            old_pos = tile["pos"].copy()

            if movement == 1: tile["pos"].y -= move_speed # Up
            elif movement == 2: tile["pos"].y += move_speed # Down
            elif movement == 3: tile["pos"].x -= move_speed # Left
            elif movement == 4: tile["pos"].x += move_speed # Right
            
            tile["pos"].x = max(0, min(self.SCREEN_WIDTH - tile["rect"].width, tile["pos"].x))
            tile["pos"].y = max(0, min(self.SCREEN_HEIGHT - tile["rect"].height, tile["pos"].y))
            tile["rect"].topleft = (int(tile["pos"].x), int(tile["pos"].y))

            if movement != 0:
                old_dist = abs(old_pos.y - self.sentence_area.centery)
                new_dist = abs(tile["pos"].y - self.sentence_area.centery)
                if new_dist < old_dist:
                    reward += 0.01

            is_now_placed = self.sentence_area.colliderect(tile["rect"])
            if is_now_placed and not tile["is_placed"]:
                reward += 0.1
                # sound: tile_place.wav
            tile["is_placed"] = is_now_placed
            
            self._align_placed_tiles()

        # Handle sentence submission
        if shift_pressed:
            submission_reward, correct = self._handle_submission()
            reward += submission_reward
            if correct:
                self.sentences_solved += 1
                if self.sentences_solved < self.SENTENCES_TO_WIN:
                    self._setup_new_puzzle()
                else:
                    self.game_over = True
                    reward += 50
                    # sound: game_win.wav

        # Update timers and step counter
        self.time_remaining -= 1
        self.steps += 1
        if self.feedback_flash:
            _, timer = self.feedback_flash
            timer -= 1
            if timer <= 0:
                self.feedback_flash = None
            else:
                self.feedback_flash = (self.feedback_flash[0], timer)
        
        # Check termination conditions
        terminated = self.game_over or self.time_remaining <= 0 or self.steps >= self.TIME_LIMIT_SECONDS * self.FPS * 1.1
        if terminated and not self.game_over:
            # sound: game_lose.wav
            pass

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _align_placed_tiles(self):
        placed_tiles = sorted(
            [t for t in self.word_tiles if t["is_placed"]], 
            key=lambda t: t["pos"].x
        )
        
        current_x = self.sentence_area.left + 10
        for tile in placed_tiles:
            tile["pos"].x = current_x
            tile["pos"].y = self.sentence_area.centery - tile["rect"].height / 2
            tile["rect"].topleft = (int(tile["pos"].x), int(tile["pos"].y))
            current_x += tile["rect"].width + 5

    def _handle_submission(self):
        placed_tiles = sorted(
            [t for t in self.word_tiles if t["is_placed"]], 
            key=lambda t: t["pos"].x
        )
        
        if not placed_tiles:
            return 0, False
            
        submitted_sentence = " ".join([t["text"] for t in placed_tiles])
        
        if submitted_sentence == self.current_sentence_text:
            # sound: correct_answer.wav
            self.feedback_flash = (self.COLOR_FEEDBACK_CORRECT, self.FPS // 2)
            reward = 1.0 + len(placed_tiles) / 2.0
            return reward, True
        else:
            # sound: incorrect_answer.wav
            self.feedback_flash = (self.COLOR_FEEDBACK_INCORRECT, self.FPS // 2)
            return -1.0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.sentence_area, border_radius=10)
        
        for i, tile in enumerate(self.word_tiles):
            color = self.COLOR_TILE_PLACED if tile["is_placed"] else self.COLOR_TILE
            pygame.draw.rect(self.screen, color, tile["rect"], border_radius=5)
            
            if self.selected_tile_index == i:
                pygame.draw.rect(self.screen, self.COLOR_TILE_SELECTED, tile["rect"], 3, border_radius=5)
            
            text_rect = tile["surf"].get_rect(center=tile["rect"].center)
            self.screen.blit(tile["surf"], text_rect)

        if self.feedback_flash:
            color, _ = self.feedback_flash
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (0, 0))

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        solved_text = self.font_ui.render(f"Sentences: {self.sentences_solved} / {self.SENTENCES_TO_WIN}", True, self.COLOR_UI_TEXT)
        text_rect = solved_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        text_rect.top = 10
        self.screen.blit(solved_text, text_rect)
        
        time_ratio = self.time_remaining / (self.TIME_LIMIT_SECONDS * self.FPS)
        time_bar_width = (self.SCREEN_WIDTH - 20) * max(0, time_ratio)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (10, 45, time_bar_width, 10), border_radius=5)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_steps": self.time_remaining,
            "sentences_solved": self.sentences_solved,
        }

    def close(self):
        pygame.font.quit()
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