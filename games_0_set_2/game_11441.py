import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:02:09.517154
# Source Brief: brief_01441.md
# Brief Index: 1441
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a timed math puzzle game.

    The player must solve increasingly difficult arithmetic problems against the clock.
    The goal is to achieve the highest score by answering correctly and quickly.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement of the selector on the on-screen numpad.
      (0=None, 1=Up, 2=Down, 3=Left, 4=Right)
    - action[1]: 'Space' button. Presses the selected numpad key.
      (0=Released, 1=Held)
    - action[2]: 'Shift' button. Clears the current answer input.
      (0=Released, 1=Held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game screen.

    Reward Structure:
    - Correct Answer: +10 * level
    - Incorrect Answer / Timeout: -5

    Termination:
    - Player runs out of lives (5 incorrect answers).
    - Episode reaches max steps (5000).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Solve arithmetic problems against the clock. Input answers using the on-screen numpad to score points and advance to harder levels."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector on the numpad. Press space to input the selected number/symbol and shift to clear your answer."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    INITIAL_LIVES = 5

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_TEXT = (220, 220, 255)
    COLOR_INPUT = (255, 255, 255)
    COLOR_PLACEHOLDER = (100, 100, 120)
    COLOR_CORRECT = (100, 255, 100)
    COLOR_INCORRECT = (255, 100, 100)
    COLOR_SELECTOR = (50, 150, 255)
    COLOR_TIMER_BAR = (50, 200, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Fonts ---
        try:
            self.font_large = pygame.font.Font(pygame.font.match_font('consolas, dejavusansmono'), 64)
            self.font_medium = pygame.font.Font(pygame.font.match_font('consolas, dejavusansmono'), 32)
            self.font_small = pygame.font.Font(pygame.font.match_font('consolas, dejavusansmono'), 24)
        except:
            self.font_large = pygame.font.Font(None, 80)
            self.font_medium = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 30)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.lives = 0
        self.game_over = False
        self.equation_str = ""
        self.correct_answer_str = ""
        self.player_input_str = ""
        self.time_remaining = 0.0
        self.equation_time_limit = 10.0
        
        # --- Input & Control State ---
        self.numpad_layout = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['-', '0', '.']]
        self.selector_pos = [0, 0]  # [row, col]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        self.move_cooldown_max = 4 # frames

        # --- Visual Effects State ---
        self.feedback_color = (0, 0, 0, 0)
        self.feedback_timer = 0
        self.feedback_duration = 10 # frames

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.player_input_str = ""
        self.selector_pos = [0, 0]
        self.feedback_timer = 0
        
        self._generate_equation()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.move_cooldown = max(0, self.move_cooldown - 1)
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # 1. Movement Action
        if self.move_cooldown == 0:
            moved = False
            if movement == 1: # Up
                self.selector_pos[0] = (self.selector_pos[0] - 1) % len(self.numpad_layout)
                moved = True
            elif movement == 2: # Down
                self.selector_pos[0] = (self.selector_pos[0] + 1) % len(self.numpad_layout)
                moved = True
            elif movement == 3: # Left
                self.selector_pos[1] = (self.selector_pos[1] - 1) % len(self.numpad_layout[0])
                moved = True
            elif movement == 4: # Right
                self.selector_pos[1] = (self.selector_pos[1] + 1) % len(self.numpad_layout[0])
                moved = True
            if moved:
                self.move_cooldown = self.move_cooldown_max

        # 2. Space Action (Select Digit)
        if space_press:
            # # SFX: Button press
            selected_char = self.numpad_layout[self.selector_pos[0]][self.selector_pos[1]]
            
            # Handle negative sign logic
            if selected_char == '-':
                if not self.player_input_str:
                    self.player_input_str += '-'
            else:
                 self.player_input_str += selected_char
        
        # 3. Shift Action (Clear Input)
        if shift_press:
            # # SFX: Clear sound
            self.player_input_str = ""

        # --- Update Game Logic ---
        self.time_remaining -= 1.0 / self.FPS

        # Check for Answer Submission (implicit by length)
        submitted = len(self.player_input_str) > 0 and \
                    (self.player_input_str == '-' and len(self.correct_answer_str) == 1) or \
                    (len(self.player_input_str) == len(self.correct_answer_str) and '-' not in self.player_input_str) or \
                    (len(self.player_input_str) == len(self.correct_answer_str) + 1 and self.player_input_str.startswith('-'))

        if submitted:
            if self.player_input_str == self.correct_answer_str:
                # # SFX: Correct answer
                reward = 10 * self.level
                self.score += int(10 * self.level * max(0.1, self.time_remaining / self.equation_time_limit))
                self.level += 1
                self._trigger_feedback(correct=True)
            else:
                # # SFX: Incorrect answer
                reward = -5
                self.lives -= 1
                self._trigger_feedback(correct=False)
            self._generate_equation()

        # Check for Timeout
        elif self.time_remaining <= 0:
            # # SFX: Timeout buzzer
            reward = -5
            self.lives -= 1
            self._trigger_feedback(correct=False)
            self._generate_equation()

        # --- Update Termination ---
        terminated = self.lives <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            self.game_over = True
            # # SFX: Game over
            if self.lives <= 0:
                reward = -5 # Final penalty

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_equation(self):
        self.equation_time_limit = max(5.0, 10.0 - 0.1 * (self.level - 1))
        self.time_remaining = self.equation_time_limit
        self.player_input_str = ""

        op_choices = ['+']
        if self.level >= 2: op_choices.append('-')
        if self.level >= 5: op_choices.append('*')
        if self.level >= 9: op_choices.append('/')

        op = self.np_random.choice(op_choices)
        
        if self.level < 13: # Two operands
            max_val = min(9 + self.level // 2, 99)
            a = self.np_random.integers(1, max_val + 1)
            b = self.np_random.integers(1, max_val + 1)

            if op == '+':
                answer = a + b
                self.equation_str = f"{a} + {b}"
            elif op == '-':
                # Ensure positive result for early levels
                if self.level < 7:
                    a, b = max(a, b), min(a, b)
                answer = a - b
                self.equation_str = f"{a} - {b}"
            elif op == '*':
                a = self.np_random.integers(2, 10 + self.level // 3)
                b = self.np_random.integers(2, 10)
                answer = a * b
                self.equation_str = f"{a} * {b}"
            else: # op == '/'
                divisor = self.np_random.integers(2, 10)
                multiplier = self.np_random.integers(2, 10 + self.level // 3)
                dividend = divisor * multiplier
                answer = multiplier
                self.equation_str = f"{dividend} / {divisor}"
        
        else: # Three operands
            max_val = min(15 + self.level // 3, 50)
            a = self.np_random.integers(1, max_val)
            b = self.np_random.integers(1, max_val)
            c = self.np_random.integers(1, max_val)
            
            # Simple left-to-right evaluation
            answer = eval(f"{a}{op}{b}{op}{c}")
            self.equation_str = f"{a} {op} {b} {op} {c}"

        self.correct_answer_str = str(int(answer))

    def _trigger_feedback(self, correct):
        if correct:
            self.feedback_color = self.COLOR_CORRECT
        else:
            self.feedback_color = self.COLOR_INCORRECT
        self.feedback_timer = self.feedback_duration

    def _render_text(self, text, font, color, center_pos, alpha=255):
        text_surf = font.render(text, True, color)
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        # --- Draw Background Gradient ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        # --- Render Feedback Flash ---
        if self.feedback_timer > 0:
            alpha = int(128 * (self.feedback_timer / self.feedback_duration))
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.feedback_color, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.feedback_timer -= 1
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Equation ---
        self._render_text(self.equation_str, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, 100))
        
        # --- Answer Input ---
        display_input = self.player_input_str
        placeholders = "_" * (len(self.correct_answer_str) - len(display_input) + (1 if display_input.startswith('-') else 0))
        
        input_surf = self.font_large.render(display_input, True, self.COLOR_INPUT)
        placeholder_surf = self.font_large.render(placeholders, True, self.COLOR_PLACEHOLDER)
        
        total_width = input_surf.get_width() + placeholder_surf.get_width()
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        
        self.screen.blit(input_surf, (start_x, 160))
        self.screen.blit(placeholder_surf, (start_x + input_surf.get_width(), 160))

        # --- Numpad ---
        numpad_w, numpad_h = 240, 220
        numpad_x, numpad_y = (self.SCREEN_WIDTH - numpad_w) // 2, 240
        cell_w, cell_h = numpad_w / 3, numpad_h / 4

        for r, row in enumerate(self.numpad_layout):
            for c, char in enumerate(row):
                char_x = numpad_x + c * cell_w + cell_w / 2
                char_y = numpad_y + r * cell_h + cell_h / 2
                self._render_text(char, self.font_medium, self.COLOR_TEXT, (char_x, char_y))
        
        # --- Numpad Selector ---
        sel_r, sel_c = self.selector_pos
        sel_x = numpad_x + sel_c * cell_w
        sel_y = numpad_y + sel_r * cell_h
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        alpha = int(100 + pulse * 100)
        
        selector_rect = pygame.Rect(sel_x, sel_y, cell_w, cell_h)
        shape_surf = pygame.Surface(selector_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_SELECTOR, alpha), shape_surf.get_rect(), border_radius=10)
        self.screen.blit(shape_surf, selector_rect)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 2, border_radius=10)

    def _render_ui(self):
        # --- Score ---
        self._render_text(f"SCORE: {self.score}", self.font_small, self.COLOR_TEXT, (100, 30))
        
        # --- Level ---
        self._render_text(f"LEVEL: {self.level}", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 100, 30))
        
        # --- Lives ---
        life_radius = 8
        life_spacing = 25
        for i in range(self.lives):
            x = 40 + i * life_spacing
            y = self.SCREEN_HEIGHT - 30
            pygame.gfxdraw.filled_circle(self.screen, x, y, life_radius, self.COLOR_CORRECT)
            pygame.gfxdraw.aacircle(self.screen, x, y, life_radius, self.COLOR_TEXT)

        # --- Timer Bar ---
        timer_width = self.SCREEN_WIDTH * 0.8
        timer_x = (self.SCREEN_WIDTH - timer_width) / 2
        
        time_ratio = max(0, self.time_remaining / self.equation_time_limit)
        current_width = int(timer_width * time_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, (timer_x, 60, timer_width, 10), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (timer_x, 60, current_width, 10), border_radius=5)

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", self.font_large, self.COLOR_INCORRECT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 40))
            self._render_text(f"FINAL SCORE: {self.score}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Math Racer")
    clock = pygame.time.Clock()
    
    total_reward = 0
    total_steps = 0
    
    # Restore the original SDL_VIDEODRIVER for local rendering
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    while not done:
        # --- Action Mapping for Human Play ---
        movement, space, shift = 0, 0, 0
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
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                total_steps = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1
        
        # --- Render to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {total_steps}")
            # Wait for 'R' to restart or quit
            restart = False
            while not restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        restart = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                        total_reward = 0
                        total_steps = 0
                        restart = True
                        break
            if done: # If we didn't restart, break the main loop
                break

    env.close()