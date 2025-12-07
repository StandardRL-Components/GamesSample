import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:50:55.397157
# Source Brief: brief_03341.md
# Brief Index: 3341
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

    The player must solve procedurally generated arithmetic problems against the clock.
    Input is handled via a virtual keypad controlled by the agent.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for keypad cursor.
    - actions[1]: Space button (0=released, 1=held) for selecting keypad entry.
    - actions[2]: Shift button (0=released, 1=held) for backspace.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game screen.

    Reward Structure:
    - +5 for a correct answer.
    - -2 for an incorrect answer.
    - +0.1 for each digit entered that correctly matches the prefix of the answer.
    - +50 terminal reward for winning (reaching the score goal).
    - -10 terminal penalty for running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Solve procedurally generated arithmetic problems against the clock. "
        "Use the virtual keypad to enter answers and score points before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor on the keypad. "
        "Press space to select a number or command (S=Submit, C=Clear). Press shift for backspace."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 50
    TIME_LIMIT_SECONDS = 60
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PLAYER_INPUT = (255, 255, 255)
    COLOR_KEYPAD = (50, 70, 90)
    COLOR_KEYPAD_TEXT = (200, 200, 220)
    COLOR_HIGHLIGHT = (100, 150, 255)
    COLOR_CORRECT = (100, 255, 100)
    COLOR_INCORRECT = (255, 100, 100)
    COLOR_TIMER_WARN = (255, 200, 0)
    
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
        self.font_large = pygame.font.Font(None, 80)
        self.font_medium = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # --- Game State ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_limit_frames = 0
        self.difficulty_level = 0
        self.correct_answers_count = 0
        self.player_input_string = ""
        self.cursor_pos = [0, 0]
        self.last_space_held = False
        self.last_shift_held = False
        self.problem = {}
        self.feedback_animation = None
        self.particles = []
        
        self.keypad_layout = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['C', '0', 'S'] # Clear, Zero, Submit
        ]
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_limit_frames = self.TIME_LIMIT_SECONDS * self.FPS
        self.difficulty_level = 0
        self.correct_answers_count = 0
        
        self.player_input_string = ""
        self.cursor_pos = [0, 0]
        self.last_space_held = False
        self.last_shift_held = False
        
        self.feedback_animation = None
        self.particles = []

        self._generate_new_problem()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Update Timers and State ---
        self.steps += 1
        self.time_limit_frames -= 1
        self._update_feedback_animation()
        self._update_particles()
        
        # --- Handle Input ---
        # Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % 3
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % 4

        # Backspace (Shift)
        if shift_held and not self.last_shift_held and len(self.player_input_string) > 0:
            self.player_input_string = self.player_input_string[:-1]
            # sfx: backspace.wav

        # Keypad selection (Space)
        if space_held and not self.last_space_held:
            key = self.keypad_layout[self.cursor_pos[1]][self.cursor_pos[0]]
            if key.isdigit() and len(self.player_input_string) < 3:
                self.player_input_string += key
                if str(self.problem['answer']).startswith(self.player_input_string):
                    reward += 0.1 # Shaping reward for correct prefix
                # sfx: key_press.wav
            elif key == 'C':
                self.player_input_string = ""
                # sfx: clear.wav
            elif key == 'S':
                submission_reward = self._check_answer()
                reward += submission_reward

        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 50
            terminated = True
            # sfx: win_game.wav
        elif self.time_limit_frames <= 0:
            reward -= 10
            terminated = True
            # sfx: lose_game.wav
            
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_new_problem(self):
        max_operand = min(9 + self.difficulty_level, 99)
        op1 = self.np_random.integers(1, max_operand + 1)
        op2 = self.np_random.integers(1, max_operand + 1)
        operator = self.np_random.choice(['+', '-', '*'])

        if operator == '-' and op1 < op2:
            op1, op2 = op2, op1

        self.problem = {
            'op1': op1,
            'op2': op2,
            'operator': operator,
            'answer': eval(f"{op1} {operator} {op2}")
        }
        self.player_input_string = ""

    def _check_answer(self):
        if not self.player_input_string:
            return 0 # No penalty for empty submission

        player_answer = int(self.player_input_string)
        if player_answer == self.problem['answer']:
            self.score += 5
            self.correct_answers_count += 1
            if self.correct_answers_count > 0 and self.correct_answers_count % 5 == 0:
                self.difficulty_level += 1
            self._trigger_feedback(correct=True)
            # sfx: correct_answer.wav
            reward = 5
        else:
            self.score = max(0, self.score - 2)
            self._trigger_feedback(correct=False)
            # sfx: incorrect_answer.wav
            reward = -2
        
        self._generate_new_problem()
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, self.time_limit_frames / self.FPS),
            "difficulty_level": self.difficulty_level
        }

    # --- Rendering Methods ---

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Problem
        problem_text = f"{self.problem['op1']} {self.problem['operator']} {self.problem['op2']} ="
        text_surf = self.font_large.render(problem_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 50))
        
        # Player Input
        input_text = self.player_input_string if self.player_input_string else "_"
        # Blinking cursor effect
        if self.steps % self.FPS < self.FPS // 2 and not self.player_input_string:
             input_text = ""
        input_surf = self.font_medium.render(input_text, True, self.COLOR_PLAYER_INPUT)
        self.screen.blit(input_surf, (self.WIDTH // 2 - input_surf.get_width() // 2, 120))
        
        # Keypad
        self._render_keypad()

        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            radius = int(p['lifespan'] / 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

        # Feedback flash
        if self.feedback_animation:
            alpha = int(128 * (self.feedback_animation['timer'] / 15.0))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.feedback_animation['color'], alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_keypad(self):
        key_w, key_h = 60, 40
        pad_w = 3 * key_w + 2 * 10
        start_x = self.WIDTH // 2 - pad_w // 2
        start_y = 180

        for y, row in enumerate(self.keypad_layout):
            for x, key in enumerate(row):
                is_highlighted = (self.cursor_pos == [x, y])
                rect = pygame.Rect(start_x + x * (key_w + 10), start_y + y * (key_h + 10), key_w, key_h)
                
                color = self.COLOR_HIGHLIGHT if is_highlighted else self.COLOR_KEYPAD
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                text_surf = self.font_small.render(key, True, self.COLOR_KEYPAD_TEXT)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer
        time_left = math.ceil(max(0, self.time_limit_frames / self.FPS))
        timer_color = self.COLOR_TIMER_WARN if time_left <= 10 else self.COLOR_TEXT
        time_surf = self.font_small.render(f"TIME: {time_left}", True, timer_color)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_surf, time_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0,0))
        
        message = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
        color = self.COLOR_CORRECT if self.score >= self.WIN_SCORE else self.COLOR_INCORRECT
        
        text_surf = self.font_large.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    # --- Helper Methods ---

    def _trigger_feedback(self, correct):
        self.feedback_animation = {
            'color': self.COLOR_CORRECT if correct else self.COLOR_INCORRECT,
            'timer': 15 # frames
        }
        # Spawn particles
        center_pos = [self.WIDTH // 2, 90]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(center_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': self.feedback_animation['color'],
                'lifespan': self.np_random.integers(15, 30)
            })

    def _update_feedback_animation(self):
        if self.feedback_animation:
            self.feedback_animation['timer'] -= 1
            if self.feedback_animation['timer'] <= 0:
                self.feedback_animation = None
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Loop ---
    running = True
    # Use a different screen for display to avoid conflicts with env's internal screen
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Math Puzzle Gym Environment")
    
    total_reward = 0
    action = [0, 0, 0] # no-op, released, released

    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    clock = pygame.time.Clock()

    while running:
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()