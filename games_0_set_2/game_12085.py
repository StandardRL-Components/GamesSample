import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:04:18.861938
# Source Brief: brief_02085.md
# Brief Index: 2085
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A retro-style arcade typing game where the player must manage an automatically
    typing system. The goal is to survive for 60 seconds.

    The core mechanic is adapted to the MultiDiscrete([5, 2, 2]) action space.
    Typing happens automatically, but can introduce errors. The agent's task is
    to decide when to submit the typed word (space) and when to use backspace
    to correct an error (shift).

    - Win Condition: Survive for 60 seconds of in-game time.
    - Lose Condition: The timer reaches zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A retro arcade typing game. Type words as they appear, but watch for auto-typed errors. "
        "Submit correct words to gain time and survive."
    )
    user_guide = (
        "Controls: Use 'space' to submit the typed word and 'shift' to use backspace to correct errors."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_BAR_BG = (40, 50, 80)
    COLOR_GREEN = (50, 255, 100)
    COLOR_RED = (255, 70, 70)
    COLOR_YELLOW = (255, 220, 100)
    COLOR_WORD_GUIDE = (60, 70, 100)
    COLOR_CORRECT_CHAR = (240, 240, 255)
    
    # Game Parameters
    INITIAL_TIMER = 10.0
    MAX_TIMER = 15.0
    WIN_TIME = 60.0
    TIME_GAIN_ON_CORRECT = 1.5
    TIME_PENALTY_ON_WRONG = 2.0
    MAX_STEPS = FPS * 90  # Max episode length of 90 seconds

    # Automatic Typing Mechanics
    TYPING_DELAY_FRAMES = 5 # Frames between each character appearing
    ERROR_CHANCE = 0.10 # 10% chance of a wrong character being typed

    # UI
    FEEDBACK_DURATION = 1.0 # seconds

    # Word List (embedded to avoid file I/O)
    WORD_LIST = [
        "code", "game", "python", "agent", "learn", "reward", "action", "state", 
        "expert", "visual", "policy", "model", "future", "quest", "drive", 
        "system", "space", "pixel", "vector", "matrix", "tensor", "value", 
        "reset", "step", "render", "play", "fun", "goal", "win", "loss", 
        "time", "score", "deep", "mind", "brain", "network", "data", "train",
        "loop", "logic", "event", "frame", "rate", "speed", "level", "build"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono,monospace'), 48)
            self.font_medium = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono,monospace'), 24)
            self.font_small = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono,monospace'), 18)
        except:
            self.font_large = pygame.font.Font(None, 60)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.total_game_time_elapsed = 0.0
        self.game_over = False
        self.game_won = False
        
        self.current_word = ""
        self.typed_buffer = ""
        self.typing_cooldown = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.feedback_message = None
        self.feedback_timer = 0.0
        
        # self.reset() is called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.INITIAL_TIMER
        self.total_game_time_elapsed = 0.0
        self.game_over = False
        self.game_won = False
        
        self.typed_buffer = ""
        self.typing_cooldown = self.TYPING_DELAY_FRAMES
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.feedback_message = None
        self.feedback_timer = 0.0
        
        self._get_new_word()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        # movement = action[0] # Not used in this game
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        terminated = False
        truncated = False
        self.steps += 1
        
        # --- Update Game Time ---
        delta_time = 1.0 / self.FPS
        self.timer -= delta_time
        self.total_game_time_elapsed += delta_time
        self.feedback_timer = max(0, self.feedback_timer - delta_time)
        self.typing_cooldown = max(0, self.typing_cooldown - 1)
        
        # --- Handle Player Actions (Edge-Triggered) ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        if space_pressed:
            reward += self._handle_submission()
        if shift_pressed:
            reward += self._handle_backspace()
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Handle Automatic Typing ---
        if not self.game_over and self.typing_cooldown == 0 and len(self.typed_buffer) < len(self.current_word):
            self._auto_type_character()

        # --- Check Termination Conditions ---
        if self.timer <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            self.feedback_message = ("TIME UP!", self.COLOR_RED)
            self.feedback_timer = 999
        elif self.total_game_time_elapsed >= self.WIN_TIME:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100
            self.feedback_message = ("YOU WIN!", self.COLOR_GREEN)
            self.feedback_timer = 999
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Truncated-like termination
        
        terminated = terminated or self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_new_word(self):
        old_word = self.current_word
        while self.current_word == old_word:
            self.current_word = self.np_random.choice(self.WORD_LIST)
        self.typed_buffer = ""
        self.typing_cooldown = self.TYPING_DELAY_FRAMES

    def _handle_submission(self):
        if not self.typed_buffer:
            return 0 # No penalty for empty submission
        
        if self.typed_buffer == self.current_word:
            # --- Correct Word ---
            self.score += 1
            self.timer = min(self.MAX_TIMER, self.timer + self.TIME_GAIN_ON_CORRECT)
            self.feedback_message = ("Correct!", self.COLOR_GREEN)
            self.feedback_timer = self.FEEDBACK_DURATION
            self._get_new_word()
            # Sound: Correct Word
            return 1.0
        else:
            # --- Incorrect Word ---
            self.timer -= self.TIME_PENALTY_ON_WRONG
            self.feedback_message = ("Wrong!", self.COLOR_RED)
            self.feedback_timer = self.FEEDBACK_DURATION
            self._get_new_word()
            # Sound: Incorrect Word
            return -2.0
    
    def _handle_backspace(self):
        if len(self.typed_buffer) > 0:
            self.typed_buffer = self.typed_buffer[:-1]
            # Sound: Backspace
            return -0.05 # Small penalty for needing to correct
        return 0

    def _auto_type_character(self):
        self.typing_cooldown = self.TYPING_DELAY_FRAMES
        next_char_index = len(self.typed_buffer)
        correct_char = self.current_word[next_char_index]
        
        if self.np_random.random() < self.ERROR_CHANCE:
            # Introduce an error
            possible_chars = "abcdefghijklmnopqrstuvwxyz"
            typed_char = self.np_random.choice(list(possible_chars.replace(correct_char, '')))
        else:
            typed_char = correct_char
        
        self.typed_buffer += typed_char
        # Sound: Keystroke

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Word Guide ---
        guide_text = self.font_large.render(self.current_word, True, self.COLOR_WORD_GUIDE)
        guide_rect = guide_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
        self.screen.blit(guide_text, guide_rect)

        # --- Render Typed Buffer with Color-Coded Characters ---
        char_x_start = guide_rect.left
        for i, char in enumerate(self.typed_buffer):
            is_correct = i < len(self.current_word) and char == self.current_word[i]
            color = self.COLOR_CORRECT_CHAR if is_correct else self.COLOR_RED
            char_surf = self.font_large.render(char, True, color)
            self.screen.blit(char_surf, (char_x_start, guide_rect.top))
            char_x_start += char_surf.get_width()

        # --- Render Blinking Cursor ---
        if not self.game_over and (self.steps // (self.FPS // 2)) % 2 == 0:
            cursor_surf = self.font_large.render("_", True, self.COLOR_CORRECT_CHAR)
            self.screen.blit(cursor_surf, (char_x_start, guide_rect.top))
            
        # --- Render Feedback Message ---
        if self.feedback_timer > 0 and self.feedback_message:
            message, color = self.feedback_message
            alpha = min(255, int(255 * (self.feedback_timer / self.FEEDBACK_DURATION)))
            
            feedback_surf = self.font_medium.render(message, True, color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 50))
            self.screen.blit(feedback_surf, feedback_rect)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # --- Timer Text ---
        timer_str = f"{max(0, self.timer):.1f}"
        timer_text = self.font_medium.render(f"TIME: {timer_str}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(timer_text, timer_rect)
        
        # --- Timer Bar ---
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - 20 - bar_width
        bar_y = timer_rect.bottom + 5
        
        # Bar background
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Bar fill
        fill_ratio = max(0, self.timer / self.INITIAL_TIMER)
        fill_width = int(bar_width * fill_ratio)
        
        if fill_ratio > 0.5:
            bar_color = self.COLOR_GREEN
        elif fill_ratio > 0.25:
            bar_color = self.COLOR_YELLOW
        else:
            bar_color = self.COLOR_RED
            
        if fill_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # --- Instructions ---
        action_text = "SPACE: Submit | SHIFT: Backspace"
        instructions_surf = self.font_small.render(action_text, True, self.COLOR_UI_TEXT)
        instructions_rect = instructions_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 25))
        self.screen.blit(instructions_surf, instructions_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "total_game_time_elapsed": self.total_game_time_elapsed,
            "game_won": self.game_won,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}, expected [5, 2, 2]"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Observation shape is {test_obs.shape}, expected {(self.HEIGHT, self.WIDTH, 3)}"
        assert test_obs.dtype == np.uint8, f"Observation dtype is {test_obs.dtype}, expected uint8"
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Control Mapping ---
    # SPACE: Submit word
    # BACKSPACE: Delete last character
    
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Typing Game Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    space_down = False
    shift_down = False # Using shift for backspace in manual play

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_down = True
                if event.key == pygame.K_BACKSPACE: # Manual control uses BACKSPACE for SHIFT action
                    shift_down = True
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    done = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_down = False
                if event.key == pygame.K_BACKSPACE:
                    shift_down = False
        
        # --- Action Assembly ---
        # Action is [movement, space, shift]
        action = [0, 1 if space_down else 0, 1 if shift_down else 0]
        
        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # --- Rendering ---
        # The observation is the rendered screen, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(GameEnv.FPS)

    env.close()