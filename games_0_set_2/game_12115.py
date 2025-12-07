import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:20:37.919388
# Source Brief: brief_02115.md
# Brief Index: 2115
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent stacks falling blocks to reach a target height.

    **Gameplay:**
    - The agent controls a falling block's horizontal position and rotation.
    - The goal is to stack 50 blocks within 90 seconds.
    - The game's difficulty increases as more blocks are stacked, with the fall speed increasing.

    **Visuals:**
    - The game features a minimalist, geometric aesthetic with a dark grid background.
    - Blocks are brightly colored for high contrast and clarity.
    - All movements and rotations are smoothly interpolated for a polished visual experience.
    - The UI clearly displays the score, timer, and number of blocks stacked.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (0=None, 1=Up(N/A), 2=Down(N/A), 3=Left, 4=Right)
    - `action[1]`: Rotate Counter-Clockwise (1=Press)
    - `action[2]`: Rotate Clockwise (1=Press)

    **Observation Space:** `Box(shape=(400, 640, 3), dtype=uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +0.01 for each block successfully stacked.
    - +2 for a "perfect" stack (well-aligned, on a stack of 2 or more).
    - -1 for an "imperfect" stack.
    - +100 for winning the game (stacking 50 blocks).
    - -10 for losing (time runs out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks to reach the target height. Earn points for precise placements, but be quick as the timer is ticking!"
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the block. Press space to rotate counter-clockwise and shift to rotate clockwise."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TARGET_LINE = (255, 255, 255)
    BLOCK_PALETTE = [
        (255, 107, 107), # Red
        (72, 219, 251), # Blue
        (29, 209, 161), # Green
        (254, 202, 87),  # Yellow
        (255, 159, 243)  # Pink
    ]

    # Game Parameters
    TIME_LIMIT_SECONDS = 90
    FPS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    BLOCK_GOAL = 50
    FLOOR_Y = 380
    BLOCK_HEIGHT = 20
    MIN_BLOCK_WIDTH = 60
    MAX_BLOCK_WIDTH = 100
    PERFECT_STACK_THRESHOLD = 5 # pixels

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.current_block = None
        self.base_fall_speed = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.base_fall_speed = 1.0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        reward += self._update_game_state()
        
        terminated = self._check_termination()
        if terminated:
            if len(self.stacked_blocks) >= self.BLOCK_GOAL:
                reward += 100 # Win bonus
            else:
                reward += -10 # Timeout penalty
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal Movement
        if movement == 3: # Left
            self.current_block['x'] -= 5
        elif movement == 4: # Right
            self.current_block['x'] += 5
        
        # Clamp horizontal position
        half_width = self.current_block['width'] / 2
        self.current_block['x'] = np.clip(self.current_block['x'], half_width, self.SCREEN_WIDTH - half_width)

        # Rotation (on press, not hold)
        # Positive angle change rotates counter-clockwise
        if space_held and not self.prev_space_held:
            self.current_block['target_angle'] += 90 # Counter-clockwise
            # sfx: rotate_sound.play()
        if shift_held and not self.prev_shift_held:
            self.current_block['target_angle'] -= 90 # Clockwise
            # sfx: rotate_sound.play()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        block = self.current_block
        
        # Smooth rotation
        angle_diff = (block['target_angle'] - block['angle'] + 180) % 360 - 180
        block['angle'] += angle_diff * 0.3 # Interpolation factor

        # Apply gravity
        block['y'] += self.base_fall_speed
        
        # Collision detection
        corners = self._get_rotated_points(block)
        min_x = min(p[0] for p in corners)
        max_x = max(p[0] for p in corners)
        lowest_y = max(p[1] for p in corners)
        
        landing_y = self.FLOOR_Y
        support_block = None

        for sb in self.stacked_blocks:
            sb_corners = self._get_rotated_points(sb)
            sb_min_x = min(p[0] for p in sb_corners)
            sb_max_x = max(p[0] for p in sb_corners)
            
            # Check for horizontal overlap
            if max(min_x, sb_min_x) < min(max_x, sb_max_x):
                # Find highest point of stacked block in overlap region
                sb_top_y = min(p[1] for p in sb_corners)
                if sb_top_y < landing_y:
                    landing_y = sb_top_y
                    support_block = sb

        # Check for landing
        if lowest_y >= landing_y:
            # Adjust final position to sit perfectly on top
            block['y'] -= (lowest_y - landing_y)
            return self._land_block(support_block)

        return 0

    def _land_block(self, support_block):
        # sfx: land_sound.play()
        self.stacked_blocks.append(self.current_block)
        
        # --- Calculate Reward ---
        reward = 0.01 # Small reward for any successful stack
        
        # Check for perfect stack
        if support_block:
            dx = abs(self.current_block['x'] - support_block['x'])
            if dx < self.PERFECT_STACK_THRESHOLD and len(self.stacked_blocks) >= 2:
                reward += 2.0 # Perfect stack bonus
                # sfx: perfect_stack_sound.play()
            else:
                reward += -1.0 # Imperfect stack penalty
        else: # Landed on floor
            reward += -1.0 # First block is always "imperfect" for this metric
        
        # --- Update Difficulty ---
        if len(self.stacked_blocks) % 10 == 0:
            self.base_fall_speed += 0.5 # Increased from 0.05 for more noticeable effect
        
        self._spawn_new_block()
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if len(self.stacked_blocks) >= self.BLOCK_GOAL:
            self.game_over = True
            return True
        return False

    def _spawn_new_block(self):
        width = self.np_random.integers(self.MIN_BLOCK_WIDTH, self.MAX_BLOCK_WIDTH + 1)
        color_index = len(self.stacked_blocks) % len(self.BLOCK_PALETTE)
        
        self.current_block = {
            'x': self.SCREEN_WIDTH / 2,
            'y': self.BLOCK_HEIGHT,
            'width': width,
            'height': self.BLOCK_HEIGHT,
            'color': self.BLOCK_PALETTE[color_index],
            'angle': 0,
            'target_angle': 0
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_target_line()
        self._render_blocks()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.FLOOR_Y), (self.SCREEN_WIDTH, self.FLOOR_Y), 2)

    def _render_target_line(self):
        target_y = self.FLOOR_Y - self.BLOCK_GOAL * self.BLOCK_HEIGHT
        if target_y > 0:
            for x in range(0, self.SCREEN_WIDTH, 20):
                start_pos = (x, target_y)
                end_pos = (x + 10, target_y)
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, start_pos, end_pos, 2)
    
    def _render_blocks(self):
        # Draw stacked blocks
        for block in self.stacked_blocks:
            self._draw_rotated_rect(block)
        # Draw current falling block
        if self.current_block:
            self._draw_rotated_rect(self.current_block)

    def _draw_rotated_rect(self, block):
        points = self._get_rotated_points(block)
        points_int = [(int(p[0]), int(p[1])) for p in points]
        
        # Use gfxdraw for anti-aliasing
        pygame.gfxdraw.aapolygon(self.screen, points_int, block['color'])
        pygame.gfxdraw.filled_polygon(self.screen, points_int, block['color'])

        # Draw outline for better contrast
        outline_color = tuple(max(0, c - 40) for c in block['color'])
        pygame.draw.lines(self.screen, outline_color, True, points_int, 2)


    def _get_rotated_points(self, block):
        x, y, w, h, angle = block['x'], block['y'], block['width'], block['height'], block['angle']
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        half_w, half_h = w / 2, h / 2
        
        corners = [
            (-half_w, -half_h), (half_w, -half_h),
            (half_w, half_h), (-half_w, half_h)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            nx = cx * cos_a - cy * sin_a + x
            ny = cx * sin_a + cy * cos_a + y
            rotated_corners.append((nx, ny))
            
        return rotated_corners

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Level / Blocks Stacked
        level_text = self.font_level.render(f"BLOCKS: {len(self.stacked_blocks)} / {self.BLOCK_GOAL}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH / 2 - level_text.get_width() / 2, self.FLOOR_Y + 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_stacked": len(self.stacked_blocks),
            "fall_speed": self.base_fall_speed
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


if __name__ == "__main__":
    # --- Manual Play Example ---
    # To run this, you'll need to unset the dummy video driver
    # e.g., run `del os.environ["SDL_VIDEODRIVER"]` before `pygame.display.set_mode`
    # Or run this script with the environment variable not set.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a display screen for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()

    while running:
        # Action defaults
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # For manual play, register press-down events for rotation
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display screen
        # Need to transpose back for pygame's native format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Blocks Stacked: {info['blocks_stacked']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()