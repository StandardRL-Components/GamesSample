import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:51:38.757592
# Source Brief: brief_00125.md
# Brief Index: 125
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Navigate a procedurally generated grid maze by rolling a die.
    The goal is to manage momentum and reach the exit before running out of rolls.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (Unused, movement is automatic)
    - actions[1]: Space button (0=released, 1=held) -> Press to roll die
    - actions[2]: Shift button (0=released, 1=held) -> Unused
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a procedurally generated maze by rolling a die. Manage your momentum to reach the exit before you run out of rolls."
    user_guide = "Press space to roll the die and move along the path. High rolls increase momentum for bigger future moves, while low rolls decrease it."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    TILE_SIZE = 32
    MAX_STEPS = 1000 # Gymnasium max steps
    MAX_ROLLS = 100 # In-game turn limit

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PATH = (80, 80, 100)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_EXIT = (255, 215, 0)
    COLOR_EXIT_GLOW = (255, 235, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_MOMENTUM_POS = (0, 255, 127) # Spring Green
    COLOR_MOMENTUM_NEG = (255, 69, 0) # OrangeRed
    COLOR_MOMENTUM_BAR_BG = (50, 50, 70)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 72, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 36, bold=True)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.grid = None
        self.solution_path = None
        self.start_pos = None
        self.exit_pos = None
        
        self.player_path_index = 0
        self.player_grid_pos = None
        self.player_pixel_pos = np.array([0.0, 0.0])
        self.player_target_pixel_pos = np.array([0.0, 0.0])
        
        self.rolls_remaining = 0
        self.momentum = 0
        self.max_momentum = 3
        
        self.last_die_roll = 0
        self.die_roll_anim_timer = 0
        
        self.prev_space_held = False
        self.is_moving = False
        
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.reset()
        
        # --- Critical Self-Check ---
        self.validate_implementation()


    def _generate_maze(self):
        """Generates a maze using randomized DFS and returns the grid and the solution path."""
        grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        visited = np.zeros_like(grid, dtype=bool)
        
        # Start DFS from a random top-row position
        start_x = self.np_random.integers(1, self.GRID_WIDTH - 1)
        start_pos = (start_x, 0)
        
        stack = deque([(start_pos, [start_pos])])
        visited[start_pos] = True
        grid[start_pos] = 1
        
        longest_path = []

        while stack:
            (cx, cy), path = stack.pop()
            
            # Found a potential exit on the bottom row
            if cy == self.GRID_HEIGHT - 1:
                if len(path) > len(longest_path):
                    longest_path = path

            neighbors = [(cx, cy - 2), (cx, cy + 2), (cx - 2, cy), (cx + 2, cy)]
            self.np_random.shuffle(neighbors)
            
            found_new = False
            for nx, ny in neighbors:
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and not visited[nx, ny]:
                    # Carve path to neighbor
                    wall_x, wall_y = (cx + nx) // 2, (cy + ny) // 2
                    grid[wall_x, wall_y] = 1
                    grid[nx, ny] = 1
                    visited[nx, ny] = True
                    stack.append(((nx, ny), path + [(nx,ny)]))
                    found_new = True
            
            # If at a dead end, backtrack to find other paths
            if not found_new and len(path) > len(longest_path) and cy == self.GRID_HEIGHT-1:
                longest_path = path

        # Ensure a valid path was found
        if not longest_path:
             # Fallback: simple straight line if generation fails
             longest_path = [(start_x, y) for y in range(self.GRID_HEIGHT)]
             for pos in longest_path:
                 grid[pos] = 1

        # Finalize path tiles on grid
        for pos in longest_path:
            grid[pos] = 1

        return grid, longest_path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid, self.solution_path = self._generate_maze()
        self.start_pos = self.solution_path[0]
        self.exit_pos = self.solution_path[-1]

        self.player_path_index = 0
        self.player_grid_pos = self.start_pos
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_target_pixel_pos = self.player_pixel_pos.copy()

        self.rolls_remaining = self.MAX_ROLLS
        self.momentum = 0
        self.last_die_roll = 0
        self.die_roll_anim_timer = 0
        
        self.prev_space_held = False
        self.is_moving = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        truncated = False # Gymnasium API requires truncated flag

        # --- Update animations and movement ---
        self._update_player_movement()
        self.die_roll_anim_timer = max(0, self.die_roll_anim_timer - 1)

        # --- Handle Actions ---
        space_held = action[1] == 1
        space_press = space_held and not self.prev_space_held

        if space_press and not self.is_moving and self.rolls_remaining > 0:
            # --- Roll Die and Move ---
            # Sound: Dice roll
            self.rolls_remaining -= 1
            
            dist_before = len(self.solution_path) - 1 - self.player_path_index
            
            roll = self.np_random.integers(1, 7)
            self.last_die_roll = roll
            self.die_roll_anim_timer = 45 # 1.5s at 30fps
            
            # Update momentum
            if roll > 3: self.momentum = min(self.max_momentum, self.momentum + 1)
            elif roll < 4: self.momentum = max(-self.max_momentum, self.momentum - 1)
            # Sound: Momentum up/down
            
            # Calculate total move distance
            move_distance = max(1, roll + self.momentum)
            
            # Update player position on path
            self.player_path_index = min(len(self.solution_path) - 1, self.player_path_index + move_distance)
            self.player_grid_pos = self.solution_path[self.player_path_index]
            self.player_target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
            self.is_moving = True
            
            # Calculate distance-based reward
            dist_after = len(self.solution_path) - 1 - self.player_path_index
            reward += (dist_before - dist_after) * 0.1 # Reward for progress
        
        elif space_press and (self.is_moving or self.rolls_remaining <= 0):
            # Sound: Error/buzz
            pass

        self.prev_space_held = space_held
        
        # --- Check for termination conditions ---
        if self.player_grid_pos == self.exit_pos and not self.is_moving:
            # Sound: Victory fanfare
            reward += 100
            terminated = True
            self.game_over = True
        elif self.rolls_remaining <= 0 and not self.is_moving:
            # Sound: Failure sound
            reward -= 10 # Penalty for running out of time
            terminated = True
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_movement(self):
        if not self.is_moving:
            return
            
        move_vec = self.player_target_pixel_pos - self.player_pixel_pos
        dist = np.linalg.norm(move_vec)

        if dist < 1.0:
            self.player_pixel_pos = self.player_target_pixel_pos.copy()
            self.is_moving = False
        else:
            # Interpolate for smooth movement
            self.player_pixel_pos += move_vec * 0.15

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rolls_remaining": self.rolls_remaining,
            "momentum": self.momentum,
            "player_pos": self.player_grid_pos,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to the center pixel of the tile."""
        offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) / 2
        offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) / 2
        px = grid_pos[0] * self.TILE_SIZE + offset_x + self.TILE_SIZE / 2
        py = grid_pos[1] * self.TILE_SIZE + offset_y + self.TILE_SIZE / 2
        return np.array([px, py])

    # --- Rendering Methods ---
    
    def _render_game(self):
        offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) / 2
        offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) / 2
        
        # Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.TILE_SIZE + offset_x, y * self.TILE_SIZE + offset_y, self.TILE_SIZE, self.TILE_SIZE)
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect, border_radius=4)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=4)

        # Draw exit tile
        exit_pixel_pos = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(0, 0, self.TILE_SIZE, self.TILE_SIZE)
        exit_rect.center = tuple(map(int, exit_pixel_pos))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        self._draw_star(self.screen, self.COLOR_BG, exit_rect.center, int(self.TILE_SIZE * 0.35))
        
        # Draw player
        self._draw_player()

    def _draw_player(self):
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.6)
        player_center = tuple(map(int, self.player_pixel_pos))
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 50))
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 50))

        # Player square
        player_rect = pygame.Rect(0, 0, self.TILE_SIZE, self.TILE_SIZE)
        player_rect.center = player_center
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect, 2, border_radius=4)
        
        # Draw momentum bar
        self._draw_momentum_bar(player_center)

    def _draw_momentum_bar(self, player_center):
        bar_width = 8
        bar_height = self.TILE_SIZE
        bar_x = player_center[0] - self.TILE_SIZE // 2 - bar_width - 5
        bar_y = player_center[1] - bar_height // 2
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR_BG, bg_rect, border_radius=3)
        
        if self.momentum != 0:
            momentum_ratio = self.momentum / self.max_momentum
            fill_height = int(abs(momentum_ratio) * (bar_height / 2))
            
            if self.momentum > 0:
                color = self.COLOR_MOMENTUM_POS
                fill_rect = pygame.Rect(bar_x, bar_y + bar_height / 2 - fill_height, bar_width, fill_height)
            else: # momentum < 0
                color = self.COLOR_MOMENTUM_NEG
                fill_rect = pygame.Rect(bar_x, bar_y + bar_height / 2, bar_width, fill_height)
                
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

        # Center line
        pygame.draw.line(self.screen, self.COLOR_TEXT, (bar_x, bar_y + bar_height // 2), (bar_x + bar_width, bar_y + bar_height // 2), 1)

    def _draw_star(self, surface, color, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36)
            radius = size if i % 2 == 0 else size * 0.4
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        # Rolls remaining
        rolls_text = self.font_small.render(f"ROLLS: {self.rolls_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(rolls_text, (10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))
        
        # Die roll animation
        if self.die_roll_anim_timer > 0:
            alpha = min(255, int(255 * (self.die_roll_anim_timer / 30.0)))
            scale = 1.0 + (1.0 - (self.die_roll_anim_timer / 45.0)) * 0.2 # Pop effect
            
            die_text_surf = self.font_large.render(str(self.last_die_roll), True, self.COLOR_EXIT)
            scaled_surf = pygame.transform.smoothscale_by(die_text_surf, scale)
            scaled_surf.set_alpha(alpha)
            
            text_rect = scaled_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(scaled_surf, text_rect)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_grid_pos == self.exit_pos:
                msg = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_MOMENTUM_NEG
                
            end_text = self.font_medium.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # The SDL_VIDEODRIVER must be unset to show a window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Maze")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        space_press = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_press = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            # Construct action based on input
            # action: [movement, space, shift]
            action = [0, 1 if space_press else 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}. Rolls left: {info['rolls_remaining']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()