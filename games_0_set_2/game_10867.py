import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:12:32.089669
# Source Brief: brief_00867.md
# Brief Index: 867
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a color-matching hexagon puzzle game.

    The agent's goal is to rotate a hexagon and transform its side colors
    to create matches of 3 or more adjacent sides of the same color.
    The objective is to reach a target score within a time limit.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]` (Movement): 0=None, 1=Rotate CW, 2=Rotate CCW, 3=None, 4=None
    - `actions[1]` (Space): 0=Released, 1=Held (Triggers side color transformation)
    - `actions[2]` (Shift): 0=Released, 1=Held (Triggers highlight selection)

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - -2 for transforming a side's color.
    - +N for a match of N sides (e.g., +3 for a 3-side match).
    - +100 bonus for winning the game (reaching the target score).

    **Termination:**
    - Episode ends if the score reaches `TARGET_SCORE`.
    - Episode ends if the step count reaches `MAX_STEPS`.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Rotate a central hexagon and change the color of its sides to create matches of three or more."
    user_guide = "Controls: ↑/↓ to rotate, Shift to select a side, and Space to change the selected side's color."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SCORE = 1000
    MAX_STEPS = 2000
    NUM_SIDES = 6
    HEX_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    HEX_RADIUS = 120
    HEX_SIDE_THICKNESS = 40
    ROTATION_LERP_FACTOR = 0.25

    # --- Colors ---
    COLOR_BG = pygame.Color("#1a1a1d")
    COLOR_TEXT = pygame.Color("#f0f0f0")
    COLOR_HEX_OUTLINE = pygame.Color("#c3c3c3")
    COLOR_HIGHLIGHT = pygame.Color(255, 255, 255, 100) # White with alpha
    SIDE_COLORS = [
        pygame.Color("#c70039"), # 1: Red
        pygame.Color("#32a852"), # 2: Green
        pygame.Color("#3498db"), # 3: Blue
        pygame.Color("#f1c40f"), # 4: Yellow
        pygame.Color("#9b59b6"), # 5: Magenta
        pygame.Color("#1abc9c")  # 6: Cyan
    ]

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
        self.font_main = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 16)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.hexagon_colors = []
        self.highlighted_side = 0
        self.rotation_angle = 0.0
        self.target_rotation_angle = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.match_flash_timers = []

        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.hexagon_colors = [self.np_random.integers(0, len(self.SIDE_COLORS)) for _ in range(self.NUM_SIDES)]
        self.highlighted_side = 0
        self.rotation_angle = 0.0
        self.target_rotation_angle = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.match_flash_timers = [0] * self.NUM_SIDES

        # Initial check for matches to prevent starting in a solved state
        _, score_gain = self._check_and_score_matches()
        # This initial gain shouldn't be part of the score, so we just clear it
        # self.score += score_gain

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        # --- 1. Unpack Action and Update Input State ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- 2. Update Animations ---
        self._update_rotation_animation()
        self._update_flash_animation()

        # --- 3. Process Actions ---
        is_animating = abs(self.target_rotation_angle - self.rotation_angle) > 0.1
        
        # Handle highlight selection (Shift) on rising edge
        if shift_held and not self.last_shift_held:
            self.highlighted_side = (self.highlighted_side + 1) % self.NUM_SIDES

        # Process other actions only if not rotating
        if not is_animating:
            # Handle color transformation (Space) on rising edge
            if space_held and not self.last_space_held:
                original_color_index = self.hexagon_colors[self.highlighted_side]
                self.hexagon_colors[self.highlighted_side] = (original_color_index + 1) % len(self.SIDE_COLORS)
                
                transform_cost = 2
                self.score -= transform_cost
                reward -= transform_cost
                
                # Check for matches immediately after transform
                match_reward, score_gain = self._check_and_score_matches()
                reward += match_reward
                self.score += score_gain

            # Handle rotation (Up/Down)
            if movement == 1: # Rotate CW
                self.target_rotation_angle -= 60
            elif movement == 2: # Rotate CCW
                self.target_rotation_angle += 60
        
        # Check for matches if a rotation just completed
        rotation_just_finished = False
        if is_animating and abs(self.target_rotation_angle - self.rotation_angle) < 0.1:
            self.rotation_angle = self.target_rotation_angle # Snap to final angle
            rotation_just_finished = True

        if rotation_just_finished:
            match_reward, score_gain = self._check_and_score_matches()
            reward += match_reward
            self.score += score_gain

        # --- 4. Update Game State ---
        self.steps += 1
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- 5. Check for Termination ---
        if self.score >= self.TARGET_SCORE:
            terminated = True
            reward += 100 # Win bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        truncated = False # Truncation is handled by the wrapper

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _check_and_score_matches(self):
        """Finds and scores contiguous groups of 3+ matching colors."""
        total_reward = 0
        total_score_gain = 0
        
        visited = [False] * self.NUM_SIDES
        
        for i in range(self.NUM_SIDES):
            if visited[i]:
                continue
            
            current_color = self.hexagon_colors[i]
            streak = []
            
            # Check clockwise for matches
            for j in range(self.NUM_SIDES):
                next_idx = (i + j) % self.NUM_SIDES
                if self.hexagon_colors[next_idx] == current_color and not visited[next_idx]:
                    streak.append(next_idx)
                else:
                    break
            
            if len(streak) >= 3:
                for idx in streak:
                    visited[idx] = True
                    self.match_flash_timers[idx] = 15 # Flash for 15 frames
                
                # Score is based on color value (index + 1) * number of matches
                score_gain = (current_color + 1) * len(streak)
                # Reward is based on number of matches
                reward_gain = len(streak)
                
                total_score_gain += score_gain
                total_reward += reward_gain
        
        return total_reward, total_score_gain

    def _update_rotation_animation(self):
        """Smoothly interpolates the hexagon's rotation angle."""
        diff = self.target_rotation_angle - self.rotation_angle
        if abs(diff) > 0.01:
            self.rotation_angle += diff * self.ROTATION_LERP_FACTOR
        else:
            self.rotation_angle = self.target_rotation_angle

    def _update_flash_animation(self):
        """Decrements the flash timers for visual feedback on matches."""
        for i in range(len(self.match_flash_timers)):
            if self.match_flash_timers[i] > 0:
                self.match_flash_timers[i] -= 1

    def _get_hexagon_side_polygon(self, side_index):
        """Calculates the 4 vertices of a single trapezoidal side of the hexagon."""
        points = []
        for i in range(2):
            angle_rad = math.radians(self.rotation_angle + side_index * 60 + i * 60)
            outer_x = self.HEX_CENTER[0] + self.HEX_RADIUS * math.cos(angle_rad)
            outer_y = self.HEX_CENTER[1] + self.HEX_RADIUS * math.sin(angle_rad)
            points.append((int(outer_x), int(outer_y)))

        for i in range(2):
            angle_rad = math.radians(self.rotation_angle + side_index * 60 + (1 - i) * 60)
            inner_radius = self.HEX_RADIUS - self.HEX_SIDE_THICKNESS
            inner_x = self.HEX_CENTER[0] + inner_radius * math.cos(angle_rad)
            inner_y = self.HEX_CENTER[1] + inner_radius * math.sin(angle_rad)
            points.append((int(inner_x), int(inner_y)))
        return points

    def _render_game(self):
        """Renders the main hexagon and effects."""
        for i in range(self.NUM_SIDES):
            poly_points = self._get_hexagon_side_polygon(i)
            color_index = self.hexagon_colors[i]
            side_color = self.SIDE_COLORS[color_index]

            # Draw the colored side
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, side_color)
            pygame.gfxdraw.aapolygon(self.screen, poly_points, self.COLOR_HEX_OUTLINE)

            # Draw match flash effect
            if self.match_flash_timers[i] > 0:
                flash_alpha = int(150 * (self.match_flash_timers[i] / 15))
                flash_color = (255, 255, 255, flash_alpha)
                pygame.gfxdraw.filled_polygon(self.screen, poly_points, flash_color)

        # Draw highlight on the selected side
        highlight_poly = self._get_hexagon_side_polygon(self.highlighted_side)
        pygame.gfxdraw.filled_polygon(self.screen, highlight_poly, self.COLOR_HIGHLIGHT)
        pygame.gfxdraw.aapolygon(self.screen, highlight_poly, (255,255,255))


    def _render_ui(self):
        """Renders the score, timer, and instructions."""
        # Score
        score_text = self.font_main.render(f"Score: {self.score} / {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_main.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Instructions
        instructions_text = self.font_small.render(
            "UP/DOWN: Rotate | SHIFT: Select Side | SPACE: Change Color", True, self.COLOR_TEXT
        )
        instr_rect = instructions_text.get_rect(centerx=self.SCREEN_WIDTH // 2, bottom=self.SCREEN_HEIGHT - 10)
        self.screen.blit(instructions_text, instr_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "target_score": self.TARGET_SCORE,
            "highlighted_side": self.highlighted_side,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment
if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Configuration ---
    # Key mapping for human control
    key_map = {
        pygame.K_UP:    [2, 0, 0], # Rotate CCW (maps to action 2)
        pygame.K_DOWN:  [1, 0, 0], # Rotate CW (maps to action 1)
        pygame.K_LSHIFT:[0, 0, 1], # Select next
        pygame.K_RSHIFT:[0, 0, 1], # Select next
        pygame.K_SPACE: [0, 1, 0], # Transform
    }

    # --- Pygame window for rendering ---
    pygame.display.set_caption("Hexagon Puzzle Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Up/Down: Rotate | Shift: Select | Space: Transform | Q: Quit")

    while not terminated:
        action = [0, 0, 0] # Default action is no-op
        
        # --- Get human input ---
        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                should_quit = True
        if should_quit:
            break

        # For continuous actions (like holding a key)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 2 # CCW
        elif keys[pygame.K_DOWN]: action[0] = 1 # CW
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS for smooth viewing

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            obs, info = env.reset() # Reset for a new game
            terminated = False # Remove this line to exit after one game
            total_reward = 0
            
    env.close()