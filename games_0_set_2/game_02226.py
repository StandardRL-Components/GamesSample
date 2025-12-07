
# Generated: 2025-08-27T19:40:45.214018
# Source Brief: brief_02226.md
# Brief Index: 2226

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist minesweeper. Reveal all safe tiles to win, but avoid the mines!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = (10, 10)
    NUM_MINES = 15
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (34, 40, 49)  # #222831
    COLOR_TILE_HIDDEN = (57, 62, 70)  # #393E46
    COLOR_TILE_REVEALED = (75, 83, 94) # #4B535E
    COLOR_GRID_LINES = (75, 83, 94)
    COLOR_CURSOR = (0, 173, 181)  # #00ADB5
    COLOR_MINE = (214, 52, 71)  # #D63447
    COLOR_TEXT = (238, 238, 238)  # #EEEEEE
    COLOR_OVERLAY = (0, 0, 0, 180)
    
    NUM_COLORS = [
        COLOR_TILE_REVEALED, # 0
        (88, 133, 175),   # 1 - Blue
        (87, 145, 101),   # 2 - Green
        (198, 93, 85),    # 3 - Red
        (66, 75, 125),    # 4 - Dark Blue
        (128, 62, 55),    # 5 - Maroon
        (75, 142, 140),   # 6 - Teal
        (0, 0, 0),        # 7 - Black
        (128, 128, 128)   # 8 - Gray
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width, self.screen_height = 640, 400
        self.grid_width, self.grid_height = self.GRID_SIZE

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_main = pygame.font.Font(None, 28)
        self.font_tile = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Calculate grid rendering properties
        self.grid_area_height = self.screen_height - 60 # Leave space for UI
        self.tile_size = min(
            self.screen_width // self.grid_width,
            self.grid_area_height // self.grid_height
        )
        self.grid_render_width = self.tile_size * self.grid_width
        self.grid_render_height = self.tile_size * self.grid_height
        self.grid_offset_x = (self.screen_width - self.grid_render_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_render_height) // 2 + 20

        # Initialize state variables
        self.true_grid = None
        self.visible_grid = None
        self.cursor_pos = None
        self.safe_tiles_to_reveal = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        self.cursor_pos = [self.grid_width // 2, self.grid_height // 2]
        self.safe_tiles_to_reveal = self.grid_width * self.grid_height - self.NUM_MINES

        self._place_mines_and_numbers()
        self.visible_grid = np.zeros(self.GRID_SIZE, dtype=np.int8) # 0=hidden, 1=revealed

        return self._get_observation(), self._get_info()

    def _place_mines_and_numbers(self):
        self.true_grid = np.zeros(self.GRID_SIZE, dtype=np.int8)
        
        mine_positions = self.np_random.choice(
            self.grid_width * self.grid_height, self.NUM_MINES, replace=False
        )
        
        for pos in mine_positions:
            x = pos % self.grid_width
            y = pos // self.grid_width
            self.true_grid[y, x] = -1 # -1 represents a mine

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.true_grid[y, x] == -1:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            if self.true_grid[ny, nx] == -1:
                                count += 1
                self.true_grid[y, x] = count

    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        
        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_width - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_height - 1)

        # 2. Handle Reveal Action
        if space_pressed:
            reward = self._reveal_tile(self.cursor_pos[0], self.cursor_pos[1])

        self.score += reward
        self.steps += 1
        
        # 3. Check Termination Conditions
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        if self.safe_tiles_to_reveal == 0 and not self.game_over:
            self.win = True
            terminated = True
            reward += 100
            self.score += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_tile(self, x, y):
        if self.visible_grid[y, x] == 1:
            return 0 # No reward/penalty for clicking an already revealed tile

        self.visible_grid[y, x] = 1
        tile_value = self.true_grid[y, x]

        if tile_value == -1:
            # Hit a mine
            # sfx: explosion
            self.game_over = True
            self._create_explosion(x, y)
            return -100

        # Revealed a safe tile
        # sfx: click_safe
        self.safe_tiles_to_reveal -= 1
        reward = 1.0 - 0.2 * tile_value
        
        if tile_value == 0:
            # Flood fill for empty tiles
            reward += self._flood_fill(x, y)
            
        return reward

    def _flood_fill(self, x, y):
        q = deque([(x, y)])
        fill_reward = 0
        
        while q:
            cx, cy = q.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        if self.visible_grid[ny, nx] == 0:
                            self.visible_grid[ny, nx] = 1
                            self.safe_tiles_to_reveal -= 1
                            
                            neighbor_val = self.true_grid[ny, nx]
                            fill_reward += 1.0 - 0.2 * neighbor_val
                            # sfx: click_multi
                            
                            if neighbor_val == 0:
                                q.append((nx, ny))
        return fill_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._render_particles()
        self._render_overlays()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "safe_tiles_remaining": self.safe_tiles_to_reveal
        }

    def _render_game(self):
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.tile_size,
                    self.grid_offset_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                
                if self.visible_grid[y, x] == 0:
                    # Hidden tile
                    if self.game_over and self.true_grid[y, x] == -1:
                        # Reveal un-clicked mines on game over
                        pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                        pygame.gfxdraw.filled_circle(
                            self.screen, 
                            rect.centerx, rect.centery, 
                            self.tile_size // 4, self.COLOR_MINE
                        )
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)
                else:
                    # Revealed tile
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                    tile_value = self.true_grid[y, x]
                    if tile_value == -1:
                        # The mine that was clicked
                        pygame.draw.rect(self.screen, self.COLOR_MINE, rect)
                    elif tile_value > 0:
                        num_text = self.font_tile.render(str(tile_value), True, self.NUM_COLORS[tile_value])
                        text_rect = num_text.get_rect(center=rect.center)
                        self.screen.blit(num_text, text_rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.tile_size,
            self.grid_offset_y + self.cursor_pos[1] * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.screen_width - 15, 15))
        self.screen.blit(steps_text, steps_rect)

    def _render_overlays(self):
        if self.game_over or self.win:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 200, 100) if self.win else self.COLOR_MINE
            
            text = self.font_large.render(message, True, color)
            text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            overlay.blit(text, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _create_explosion(self, grid_x, grid_y):
        center_x = self.grid_offset_x + grid_x * self.tile_size + self.tile_size // 2
        center_y = self.grid_offset_y + grid_y * self.tile_size + self.tile_size // 2
        
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 6)
            color = random.choice([self.COLOR_MINE, (255, 159, 67), (255, 204, 92)])
            self.particles.append([
                [center_x, center_y], # pos
                [vx, vy],             # vel
                radius,               # radius
                life,                 # life
                color                 # color
            ])

    def _render_particles(self):
        if not self.particles:
            return

        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[3] -= 1          # life -= 1
            
            current_radius = int(p[2] * (p[3] / 40.0))
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p[0][0]), int(p[0][1]), current_radius, p[4]
                )
        
        self.particles = [p for p in self.particles if p[3] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Minesweeper Gym Environment")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            # Since auto_advance is False, we need to step on every key press or no-op
            # To make it playable, we only step when an action is taken
            if movement != 0 or space != 0:
                action = [movement, space, 0] # shift is not used
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate for human playability

    env.close()