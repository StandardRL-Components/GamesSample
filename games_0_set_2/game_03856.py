
# Generated: 2025-08-28T00:39:01.362868
# Source Brief: brief_03856.md
# Brief Index: 3856

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield to find the exit. Each step costs points, "
        "but reaching the goal gives a huge bonus. Avoid hitting three mines!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 31, 19 # Odd numbers for wall generation
        self.CELL_SIZE = 18
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.MAZE_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.MAZE_HEIGHT * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PATH = (30, 30, 45)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 50)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_EXIT = (50, 255, 50)
        self.COLOR_UI_TEXT = (200, 200, 220)

        # Game constants
        self.MAX_STEPS = 1000
        self.MAX_LIVES = 3
        self.NUM_MINES = 25
        
        # Rewards
        self.REWARD_STEP = -0.1
        self.REWARD_MINE = -10.0
        self.REWARD_EXIT = 100.0

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont('Consolas', 20)
            self.font_title = pygame.font.SysFont('Consolas', 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_title = pygame.font.SysFont(None, 60)
        
        # State variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.mines = set()
        self.traversed_path = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.lives = 0
        self.game_over = False
        self.rng = None
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.traversed_path = []
        self.particles = []

        self._generate_maze()
        self.player_pos = [1, 1]
        self.exit_pos = [self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2]
        
        # Ensure start and exit are open, in case of generation error
        self.maze[self.player_pos[1], self.player_pos[0]] = 0
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 0

        self._place_mines()

        self.traversed_path.append(tuple(self.player_pos))
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        stack = [(1, 1)]
        self.maze[1, 1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH - 1 and 0 < ny < self.MAZE_HEIGHT - 1 and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Use Gymnasium's RNG for reproducibility
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                self.maze[ny, nx] = 0
                self.maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_mines(self):
        empty_cells = []
        for r in range(self.MAZE_HEIGHT):
            for c in range(self.MAZE_WIDTH):
                if self.maze[r, c] == 0:
                    pos = (c, r)
                    if list(pos) != self.player_pos and list(pos) != self.exit_pos:
                        empty_cells.append(pos)
        
        num_mines = min(self.NUM_MINES, len(empty_cells))
        mine_indices = self.np_random.choice(len(empty_cells), size=num_mines, replace=False)
        self.mines = {empty_cells[i] for i in mine_indices}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        moved = self._handle_movement(movement)
        
        if moved:
            # sfx: step sound
            self.steps += 1
            reward += self.REWARD_STEP
            
            player_pos_tuple = tuple(self.player_pos)
            
            if player_pos_tuple in self.mines:
                # sfx: explosion
                self.mines.remove(player_pos_tuple)
                self.lives -= 1
                reward += self.REWARD_MINE
                self._create_explosion(self.player_pos)
            
            if self.player_pos == self.exit_pos:
                # sfx: win sound
                reward += self.REWARD_EXIT
                self.game_over = True
        
        self.score += reward
        
        terminated = self.game_over or self.lives <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        px, py = self.player_pos
        next_pos = list(self.player_pos)
        
        if movement == 1: next_pos[1] -= 1  # Up
        elif movement == 2: next_pos[1] += 1 # Down
        elif movement == 3: next_pos[0] -= 1 # Left
        elif movement == 4: next_pos[0] += 1 # Right
        else: return False # No-op

        nx, ny = next_pos
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == 0:
            self.player_pos = next_pos
            if tuple(self.player_pos) not in self.traversed_path:
                self.traversed_path.append(tuple(self.player_pos))
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        to_screen = lambda x, y: (
            self.GRID_OFFSET_X + x * self.CELL_SIZE,
            self.GRID_OFFSET_Y + y * self.CELL_SIZE
        )

        for (x, y) in self.traversed_path:
            sx, sy = to_screen(x, y)
            pygame.draw.rect(self.screen, self.COLOR_PATH, (sx, sy, self.CELL_SIZE, self.CELL_SIZE))

        for r in range(self.MAZE_HEIGHT):
            for c in range(self.MAZE_WIDTH):
                if self.maze[r, c] == 1:
                    sx, sy = to_screen(c, r)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (sx, sy, self.CELL_SIZE, self.CELL_SIZE))

        ex, ey = to_screen(self.exit_pos[0], self.exit_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, self.CELL_SIZE, self.CELL_SIZE))
        if self.game_over and self.player_pos == self.exit_pos:
             s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
             s.fill((*self.COLOR_EXIT, 100))
             self.screen.blit(s, (ex, ey))

        for (mx, my) in self.mines:
            sx, sy = to_screen(mx, my)
            center_x, center_y = sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_MINE)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_MINE)

        px, py = to_screen(self.player_pos[0], self.player_pos[1])
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        radius = int(self.CELL_SIZE / 2.5)
        
        glow_radius = int(radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        lives_text = f"LIVES: {'● ' * self.lives}{'○ ' * (self.MAX_LIVES - self.lives)}"
        steps_text = f"STEPS: {self.steps}"
        
        lives_surf = self.font_ui.render(lives_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(lives_surf, (15, 10))
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 15, 10))

        if self.game_over:
            if self.player_pos == self.exit_pos:
                end_text, color = "SUCCESS!", self.COLOR_EXIT
            elif self.lives <= 0:
                end_text, color = "GAME OVER", self.COLOR_MINE
            else:
                end_text, color = "TIME OUT", self.COLOR_UI_TEXT

            end_surf = self.font_title.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surf = self.font_title.render(end_text, True, (0,0,0))
            self.screen.blit(shadow_surf, (end_rect.x+3, end_rect.y+3))
            self.screen.blit(end_surf, end_rect)

    def _create_explosion(self, grid_pos):
        sx, sy = to_screen = (
            self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        )
        center_x, center_y = sx + self.CELL_SIZE // 2, sy + self.CELL_SIZE // 2
        
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'lifespan': lifespan, 'radius': radius})

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            
            if p['lifespan'] > 0 and p['radius'] > 0.5:
                alpha = int(255 * (p['lifespan'] / 30))
                color = (*self.COLOR_MINE, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup a window to display the game
    pygame.display.set_caption("Minefield Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("--- Minefield Maze ---")
    print(env.game_description)
    print(env.user_guide)

    while not done:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # In this game, an action must be taken to advance the frame
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Lives: {info['lives']}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print("Game Over!")
            # Wait a bit before closing
            pygame.time.wait(3000)

        # Since auto_advance is False, we only need to tick the clock to handle window events
        clock.tick(30)

    env.close()