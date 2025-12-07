
# Generated: 2025-08-27T17:45:20.377151
# Source Brief: brief_01630.md
# Brief Index: 1630

        
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
    user_guide = (
        "Controls: Arrow keys to move the robot. Collect all gems before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a gem-collecting robot through a grid-based maze, strategically maneuvering around "
        "obstacles to collect all gems within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.UI_HEIGHT = 40
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = (self.SCREEN_HEIGHT - self.UI_HEIGHT) // self.CELL_SIZE
        
        self.INITIAL_MOVES = 60
        self.NUM_GEMS = 5
        self.NUM_OBSTACLES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (45, 45, 60)
        self.COLOR_ROBOT = (255, 255, 255)
        self.COLOR_ROBOT_OUTLINE = (20, 20, 20)
        self.COLOR_OBSTACLE = (80, 80, 90)
        self.GEM_COLORS = [
            (255, 80, 80),
            (80, 255, 80),
            (100, 100, 255),
            (255, 255, 80),
            (255, 80, 255),
        ]
        self.COLOR_TEXT = (220, 220, 220)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.robot_pos = (0, 0)
        self.gem_positions = []
        self.obstacle_positions = set()
        self.moves_remaining = 0
        self.gems_collected = 0
        self.total_gems = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.INITIAL_MOVES
        self.gems_collected = 0
        self.total_gems = self.NUM_GEMS

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Creates a new random level layout."""
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        self.robot_pos = all_coords.pop()
        self.obstacle_positions = {all_coords.pop() for _ in range(self.NUM_OBSTACLES)}
        self.gem_positions = [all_coords.pop() for _ in range(self.NUM_GEMS)]

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        gx, gy = grid_pos
        px = gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = gy * self.CELL_SIZE + self.CELL_SIZE // 2 + self.UI_HEIGHT
        return px, py

    def _get_dist_to_nearest_gem(self, pos):
        """Calculates Manhattan distance to the nearest gem."""
        if not self.gem_positions:
            return 0
        
        px, py = pos
        return min(abs(px - gx) + abs(py - gy) for gx, gy in self.gem_positions)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0
        
        old_pos = self.robot_pos
        dist_before = self._get_dist_to_nearest_gem(old_pos)
        
        moved = False
        if movement != 0:
            px, py = self.robot_pos
            tx, ty = px, py

            if movement == 1: ty -= 1 # Up
            elif movement == 2: ty += 1 # Down
            elif movement == 3: tx -= 1 # Left
            elif movement == 4: tx += 1 # Right

            if (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT and
                    (tx, ty) not in self.obstacle_positions):
                self.robot_pos = (tx, ty)
                self.moves_remaining -= 1
                moved = True
        
        if self.robot_pos in self.gem_positions:
            # sfx: gem_collect
            self.gem_positions.remove(self.robot_pos)
            self.gems_collected += 1
            reward += 10
            self.score += 10
            
        if moved:
            dist_after = self._get_dist_to_nearest_gem(self.robot_pos)
            if dist_before > 0:
                if dist_after < dist_before:
                    reward += 1
                elif dist_after > dist_before:
                    reward -= 0.1
        
        terminated = False
        if not self.gem_positions:
            self.game_over = True
            terminated = True
            reward += 100
            self.score += 100
        elif self.moves_remaining <= 0:
            self.game_over = True
            terminated = True
            reward -= 50
            self.score -= 50
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        """Renders the main game area (grid, entities)."""
        for x in range(0, self.SCREEN_WIDTH + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT))
        for y in range(self.UI_HEIGHT, self.SCREEN_HEIGHT + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        for ox, oy in self.obstacle_positions:
            px, py = self._grid_to_pixel((ox, oy))
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            
        for i, (gx, gy) in enumerate(self.gem_positions):
            px, py = self._grid_to_pixel((gx, gy))
            color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 2 - 4, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 2 - 4, color)

        rx, ry = self._grid_to_pixel(self.robot_pos)
        robot_rect = pygame.Rect(rx - self.CELL_SIZE//2 + 4, ry - self.CELL_SIZE//2 + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_OUTLINE, robot_rect.inflate(4, 4))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)

    def _render_ui(self):
        """Renders the UI elements (score, moves, etc.)."""
        pygame.draw.rect(self.screen, (15, 15, 25), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        gem_text = self.font_ui.render(f"Gems: {self.gems_collected} / {self.total_gems}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))
        
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(right=self.SCREEN_WIDTH - 10, top=10)
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            msg, color = ("LEVEL COMPLETE", (100, 255, 100)) if not self.gem_positions else ("OUT OF MOVES", (255, 100, 100))
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
            "moves_remaining": self.moves_remaining,
            "gems_collected": self.gems_collected,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    running = True
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        movement = 0
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r: env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE: running = False
                
                if movement != 0:
                    action_taken = True
        
        if action_taken:
            action = [movement, 0, 0]
            obs, reward, terminated, _, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()