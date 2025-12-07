import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:48:54.345405
# Source Brief: brief_03457.md
# Brief Index: 3457
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a probabilistic quantum wave puzzle game.

    The agent controls the intended direction of a "wave function" on a 10x10 grid.
    The movement is probabilistic. Colliding with walls creates ripple effects that
    penalize the agent if they become too intense. The goal is to navigate the wave
    to a target cell within a limited number of moves.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused

    **Observation Space:** A 640x400 RGB image of the game state.

    **Rewards:**
    - Reaching the goal: +100
    - Running out of moves: -20
    - Moving closer to the goal: +1 per unit of Manhattan distance decreased
    - Moving further from the goal: -1 per unit of Manhattan distance increased
    - Each step: -0.1
    - Creating a high-intensity ripple (intensity > 4): -5
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a probabilistic quantum wave to a target on a grid, avoiding "
        "high-intensity ripples created by colliding with walls."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to guide the wave. Each move is probabilistic "
        "and has a chance to go in an unintended direction."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 40  # Grid will be 400x400
    MAX_MOVES = 20
    FPS = 30  # Affects visual effect speed

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_WAVE = (0, 255, 255)  # Cyan
    COLOR_GLOW = (0, 150, 150)
    COLOR_GOAL = (50, 255, 50)  # Bright Green
    COLOR_TEXT = (220, 220, 240)
    COLOR_RIPPLE_LOW = (0, 100, 255)  # Blue
    COLOR_RIPPLE_HIGH = (255, 50, 100)  # Red

    class Ripple:
        """Helper class for managing visual ripple effects."""
        def __init__(self, grid_pos, intensity):
            self.grid_pos = grid_pos
            self.intensity = intensity
            self.max_lifetime = GameEnv.FPS * 1.5  # Lasts 1.5 seconds
            self.lifetime = self.max_lifetime
            self.max_radius = GameEnv.CELL_SIZE * 2.5

        def update(self):
            """Decrements lifetime and returns if the ripple is still active."""
            self.lifetime -= 1
            return self.lifetime > 0

        def draw(self, surface, grid_offset_x, grid_offset_y):
            """Draws the expanding ripple effect on the given surface."""
            progress = 1.0 - (self.lifetime / self.max_lifetime)
            if progress < 0: return

            center_x = grid_offset_x + self.grid_pos[0] * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE // 2
            center_y = grid_offset_y + self.grid_pos[1] * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE // 2

            current_radius = int(self.max_radius * progress)
            alpha = int(150 * (1.0 - progress**2))  # Fade out faster at the end
            if alpha <= 0: return

            # Interpolate color based on the ripple's initial intensity
            color_factor = min(1.0, (self.intensity - 2) / 8.0)
            r = int(GameEnv.COLOR_RIPPLE_LOW[0] + (GameEnv.COLOR_RIPPLE_HIGH[0] - GameEnv.COLOR_RIPPLE_LOW[0]) * color_factor)
            g = int(GameEnv.COLOR_RIPPLE_LOW[1] + (GameEnv.COLOR_RIPPLE_HIGH[1] - GameEnv.COLOR_RIPPLE_LOW[1]) * color_factor)
            b = int(GameEnv.COLOR_RIPPLE_LOW[2] + (GameEnv.COLOR_RIPPLE_HIGH[2] - GameEnv.COLOR_RIPPLE_LOW[2]) * color_factor)
            color = (np.clip(r,0,255), np.clip(g,0,255), np.clip(b,0,255), alpha)

            pygame.gfxdraw.aacircle(surface, center_x, center_y, current_radius, color)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        # Game layout
        self.grid_width = self.GRID_SIZE * self.CELL_SIZE
        self.grid_height = self.GRID_SIZE * self.CELL_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        # Initialize state variables
        self.wave_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.moves_remaining = 0
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.ripple_intensity_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.visual_ripples = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False

        # Place wave and goal randomly, ensuring they are not at the same spot
        self.wave_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
        while True:
            self.goal_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if self.goal_pos != self.wave_pos and self._manhattan_distance(self.wave_pos, self.goal_pos) > 3:
                break

        # Reset ripple effects
        self.ripple_intensity_grid.fill(0)
        self.visual_ripples = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -0.1  # Per-step penalty

        prev_dist = self._manhattan_distance(self.wave_pos, self.goal_pos)

        if movement != 0:  # 0 is no-op
            self.moves_remaining -= 1
            ripple_penalty = self._move_wave(movement)
            reward += ripple_penalty
        
        # Dissipate ripple intensity grid over time
        self.ripple_intensity_grid = np.maximum(0, self.ripple_intensity_grid - 1)
        
        # Update visual effects
        self.visual_ripples = [r for r in self.visual_ripples if r.update()]

        current_dist = self._manhattan_distance(self.wave_pos, self.goal_pos)
        reward += (prev_dist - current_dist) # Reward for getting closer

        terminated = False
        if self.wave_pos == self.goal_pos:
            reward += 100.0
            terminated = True
        elif self.moves_remaining <= 0:
            reward -= 20.0
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _move_wave(self, intended_direction):
        """Handles the probabilistic movement of the wave and wall collisions."""
        # Directions: 1=up, 2=down, 3=left, 4=right -> Deltas: (dx, dy)
        deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        directions = list(deltas.keys())

        # Determine actual movement direction based on probability
        rand_val = self.np_random.random()
        if rand_val < 0.7: # 70% chance of intended direction
            move_dir = intended_direction
        else: # 30% chance of other directions (10% each)
            other_dirs = [d for d in directions if d != intended_direction]
            sub_rand = (rand_val - 0.7) / 0.3
            if sub_rand < 1/3: move_dir = other_dirs[0]
            elif sub_rand < 2/3: move_dir = other_dirs[1]
            else: move_dir = other_dirs[2]
        
        dx, dy = deltas[move_dir]
        new_x, new_y = self.wave_pos[0] + dx, self.wave_pos[1] + dy

        collided = not (0 <= new_x < self.GRID_SIZE and 0 <= new_y < self.GRID_SIZE)
        
        ripple_penalty = 0.0
        if collided:
            final_x = np.clip(new_x, 0, self.GRID_SIZE - 1)
            final_y = np.clip(new_y, 0, self.GRID_SIZE - 1)
            
            # Check for high-intensity penalty BEFORE incrementing
            if self.ripple_intensity_grid[final_y, final_x] > 2: # Will cross 4 after adding 2
                ripple_penalty = -5.0
            
            self.ripple_intensity_grid[final_y, final_x] += 2
            self.visual_ripples.append(self.Ripple((final_x, final_y), self.ripple_intensity_grid[final_y, final_x]))
            self.wave_pos = (final_x, final_y)
        else:
            self.wave_pos = (new_x, new_y)
            
        return ripple_penalty

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_ripples()
        self._draw_goal()
        self._draw_wave()
        self._draw_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal lines
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.CELL_SIZE)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _draw_goal(self):
        gx, gy = self.goal_pos
        rect = pygame.Rect(
            self.grid_offset_x + gx * self.CELL_SIZE,
            self.grid_offset_y + gy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Draw a slightly smaller, inset square for the goal
        pygame.draw.rect(self.screen, self.COLOR_GOAL, rect.inflate(-8, -8), border_radius=4)

    def _draw_wave(self):
        wx, wy = self.wave_pos
        cx = self.grid_offset_x + wx * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.grid_offset_y + wy * self.CELL_SIZE + self.CELL_SIZE // 2

        # Pulsating glow effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # Normalized sine 0-1
        base_radius = int(self.CELL_SIZE * 0.3)
        pulse_amount = pulse * 5

        # Glow layers
        for i in range(int(pulse_amount) + 5):
            alpha = 80 * (1 - (i / (pulse_amount + 5)))
            radius = base_radius + i
            color = (*self.COLOR_GLOW, int(alpha))
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)

        # Core circle
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, base_radius, self.COLOR_WAVE)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, base_radius, self.COLOR_WAVE)

    def _draw_ripples(self):
        for ripple in self.visual_ripples:
            ripple.draw(self.screen, self.grid_offset_x, self.grid_offset_y)

    def _draw_ui(self):
        moves_text = self.font.render(f"Moves Left: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        score_text_str = f"Score: {self.score:.1f}"
        score_text = self.font.render(score_text_str, True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Quantum Wave Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue

                # Manual control mapping
                action = [0, 0, 0] # Default: no-op
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Get the observation from the environment and draw it
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame surface wants (W, H).
        # We need to transpose it back for display.
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()