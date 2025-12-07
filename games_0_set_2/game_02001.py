
# Generated: 2025-08-28T03:21:59.998478
# Source Brief: brief_02001.md
# Brief Index: 2001

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
from itertools import product
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Reach the green exit square while avoiding hidden mines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Navigate a minefield to reach the exit in the fewest steps. You have 3 lives. The number of mines increases with each success."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 8
        self.TILE_SIZE = 50
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_W * self.TILE_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_H * self.TILE_SIZE) // 2
        self.MAX_STEPS = 500
        self.INITIAL_MINE_COUNT = 10
        self.INITIAL_LIVES = 3

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_EXPLOSION = (255, 50, 50)
        self.COLOR_HEART = (220, 30, 30)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GAMEOVER_TEXT = (255, 80, 80)
        self.COLOR_WIN_TEXT = (80, 255, 80)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.Font(None, 36)
            self.font_large = pygame.font.Font(None, 72)
        except IOError:
            self.font_main = pygame.font.SysFont("Arial", 24)
            self.font_large = pygame.font.SysFont("Arial", 48)

        # --- Game State ---
        self.mine_count = self.INITIAL_MINE_COUNT
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.lives = 0
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.mine_positions = set()
        self.explosion_timer = 0
        self.explosion_pos = None

        # --- Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.explosion_timer = 0
        self.explosion_pos = None
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates a new level, ensuring a path to the exit exists."""
        all_tiles = list(product(range(self.GRID_W), range(self.GRID_H)))
        
        self.player_pos = (0, self.np_random.integers(0, self.GRID_H))
        self.exit_pos = (self.GRID_W - 1, self.np_random.integers(0, self.GRID_H))

        available_for_mines = [
            p for p in all_tiles if p not in [self.player_pos, self.exit_pos]
        ]
        
        max_mines = min(self.mine_count, len(available_for_mines))

        while True:
            mine_indices = self.np_random.choice(
                len(available_for_mines), max_mines, replace=False
            )
            self.mine_positions = {available_for_mines[i] for i in mine_indices}

            if self._is_path_possible():
                break

    def _is_path_possible(self):
        """Checks if a path exists from player to exit using Breadth-First Search."""
        q = deque([self.player_pos])
        visited = {self.player_pos}

        while q:
            x, y = q.popleft()

            if (x, y) == self.exit_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in self.mine_positions:
                        visited.add(neighbor)
                        q.append(neighbor)
        return False

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost of taking a step

        # --- Update player position ---
        px, py = self.player_pos
        moved = False
        if movement == 1:  # Up
            if py > 0: py -= 1; moved = True
        elif movement == 2:  # Down
            if py < self.GRID_H - 1: py += 1; moved = True
        elif movement == 3:  # Left
            if px > 0: px -= 1; moved = True
        elif movement == 4:  # Right
            if px < self.GRID_W - 1: px += 1; moved = True
        
        # Only update state if a move happened (or it was a no-op)
        if moved or movement == 0:
            self.steps += 1
            if moved:
                self.player_pos = (px, py)
        else: # Bumping into a wall is still a step
            self.steps += 1
            
        # --- Check for game events ---
        terminated = False
        if self.player_pos in self.mine_positions:
            # Play explosion sound
            reward -= 10.0
            self.lives -= 1
            self.mine_positions.remove(self.player_pos)
            self.explosion_pos = self.player_pos
            self.explosion_timer = 15  # frames for animation

        if self.player_pos == self.exit_pos:
            # Play win sound
            reward += 10.0
            self.game_over = True
            terminated = True
            self.mine_count += 2  # Increase difficulty for next round

        if self.lives <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            start = (self.GRID_X_OFFSET + x * self.TILE_SIZE, self.GRID_Y_OFFSET)
            end = (self.GRID_X_OFFSET + x * self.TILE_SIZE, self.GRID_Y_OFFSET + self.GRID_H * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_H + 1):
            start = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.TILE_SIZE)
            end = (self.GRID_X_OFFSET + self.GRID_W * self.TILE_SIZE, self.GRID_Y_OFFSET + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.GRID_X_OFFSET + ex * self.TILE_SIZE,
            self.GRID_Y_OFFSET + ey * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.GRID_X_OFFSET + px * self.TILE_SIZE + 5,
            self.GRID_Y_OFFSET + py * self.TILE_SIZE + 5,
            self.TILE_SIZE - 10,
            self.TILE_SIZE - 10,
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Draw explosion effect
        if self.explosion_timer > 0:
            ex, ey = self.explosion_pos
            center_x = int(self.GRID_X_OFFSET + (ex + 0.5) * self.TILE_SIZE)
            center_y = int(self.GRID_Y_OFFSET + (ey + 0.5) * self.TILE_SIZE)
            
            # Animate radius and alpha
            radius = int(self.TILE_SIZE * 0.8 * (1 - self.explosion_timer / 15))
            alpha = int(255 * (self.explosion_timer / 15))
            
            # Use gfxdraw for anti-aliased circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*self.COLOR_EXPLOSION, alpha))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*self.COLOR_EXPLOSION, alpha))
            
            self.explosion_timer -= 1


    def _render_ui(self):
        # Draw hearts for lives
        for i in range(self.lives):
            self._draw_heart(25 + i * 35, 30)

        # Draw step count
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 15))
        
        # Draw game over/win message
        if self.game_over:
            if self.player_pos == self.exit_pos:
                msg = "YOU WIN!"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER_TEXT
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, x, y):
        """Draws a heart shape for the UI."""
        points = [
            (x, y - 10), (x + 5, y - 15), (x + 10, y - 10), (x, y),
            (x - 10, y - 10), (x - 5, y - 15), (x, y-10)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "mine_count": self.mine_count
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        # We need to reset to generate a valid observation
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset(seed=1)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Minefield Navigator")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    print("\n" + "="*30)
    print("Minefield Navigator - Manual Control")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting game...")
                    obs, info = env.reset()
                    total_reward = 0.0
                    
                # For turn-based, we only care about key presses, not holds
                if not env.game_over:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # Process the action
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Lives: {info['lives']}")
                    
                    if terminated:
                        print(f"Game Over! Final Score: {info['score']:.2f}")

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play
        
    env.close()