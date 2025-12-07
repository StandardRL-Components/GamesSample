
# Generated: 2025-08-28T01:24:16.446897
# Source Brief: brief_04084.md
# Brief Index: 4084

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press spacebar on a key's location to collect it. "
        "Collect all 3 keys and reach the green exit before you run out of moves."
    )

    game_description = (
        "A top-down puzzle game. Navigate a procedurally generated maze, collect three "
        "hidden keys, and find the exit. Every action costs a move, so plan your path carefully!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 20, 12
        self.MAX_MOVES = 100
        self.MAX_STEPS = 1000 # Safety termination

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (70, 80, 100)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_PULSE = (150, 200, 255)
        self.COLOR_KEY = (255, 220, 0)
        self.COLOR_EXIT = (0, 200, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Maze Rendering Geometry ---
        self.MAZE_AREA_RECT = pygame.Rect(40, 60, 560, 300)
        self.CELL_WIDTH = self.MAZE_AREA_RECT.width / self.MAZE_WIDTH
        self.CELL_HEIGHT = self.MAZE_AREA_RECT.height / self.MAZE_HEIGHT

        # --- Game State (initialized in reset) ---
        self.maze_grid = None
        self.player_pos = None
        self.key_locations = None
        self.collected_keys = None
        self.exit_pos = None
        self.remaining_moves = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.last_action_step = -1
        self.particles = []

        self.reset()

    def _generate_maze(self):
        grid = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(self.MAZE_WIDTH)] for _ in range(self.MAZE_HEIGHT)]
        
        # Use Gymnasium's seeded RNG
        start_x, start_y = self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT)
        stack = [(start_x, start_y)]
        grid[start_y][start_x]['visited'] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not grid[cy - 1][cx]['visited']: neighbors.append((cx, cy - 1, 'N'))
            if cy < self.MAZE_HEIGHT - 1 and not grid[cy + 1][cx]['visited']: neighbors.append((cx, cy + 1, 'S'))
            if cx < self.MAZE_WIDTH - 1 and not grid[cy][cx + 1]['visited']: neighbors.append((cx + 1, cy, 'E'))
            if cx > 0 and not grid[cy][cx - 1]['visited']: neighbors.append((cx - 1, cy, 'W'))

            if neighbors:
                # np_random.choice doesn't work on lists of tuples, so we manually select an index
                idx = self.np_random.integers(len(neighbors))
                nx, ny, direction = neighbors[idx]
                
                if direction == 'N':
                    grid[cy][cx]['N'] = False
                    grid[ny][nx]['S'] = False
                elif direction == 'S':
                    grid[cy][cx]['S'] = False
                    grid[ny][nx]['N'] = False
                elif direction == 'E':
                    grid[cy][cx]['E'] = False
                    grid[ny][nx]['W'] = False
                elif direction == 'W':
                    grid[cy][cx]['W'] = False
                    grid[ny][nx]['E'] = False
                
                grid[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze_grid = self._generate_maze()
        
        possible_locs = [(x, y) for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT)]
        self.np_random.shuffle(possible_locs)
        
        self.player_pos = list(possible_locs.pop())
        self.exit_pos = list(possible_locs.pop())
        self.key_locations = [list(possible_locs.pop()) for _ in range(3)]
        self.collected_keys = []

        self.steps = 0
        self.score = 0
        self.remaining_moves = self.MAX_MOVES
        self.game_over = False
        self.win_condition = False
        self.last_action_step = -1
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        is_noop = movement == 0 and not space_pressed
        
        if not self.game_over and not is_noop:
            self.remaining_moves -= 1
            reward -= 0.1
            self.last_action_step = self.steps

            # --- Handle Movement ---
            if movement > 0:
                px, py = self.player_pos
                cell = self.maze_grid[py][px]
                if movement == 1 and not cell['N']: self.player_pos[1] -= 1 # Up
                elif movement == 2 and not cell['S']: self.player_pos[1] += 1 # Down
                elif movement == 3 and not cell['W']: self.player_pos[0] -= 1 # Left
                elif movement == 4 and not cell['E']: self.player_pos[0] += 1 # Right

            # --- Handle Key Collection ---
            if space_pressed:
                for key_pos in self.key_locations:
                    if self.player_pos == key_pos and key_pos not in self.collected_keys:
                        self.collected_keys.append(key_pos)
                        reward += 10.0
                        self.score += 10
                        self._spawn_particles(self.player_pos, self.COLOR_KEY)
                        # Sound effect: KEY_COLLECT
                        break
        
        # --- Check Termination Conditions ---
        if not self.game_over:
            if self.player_pos == self.exit_pos:
                if len(self.collected_keys) == 3:
                    reward += 100.0
                    self.score += 100
                    self.win_condition = True
                    # Sound effect: WIN
                else:
                    reward -= 50.0
                    self.score -= 50
                    # Sound effect: FAIL
                terminated = True
                self.game_over = True

            elif self.remaining_moves <= 0:
                terminated = True
                self.game_over = True
                # Sound effect: LOSE
        
        if self.steps >= self.MAX_STEPS -1:
             terminated = True
             self.game_over = True

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Update and Render Particles ---
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)
            else:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
                size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
                if size > 0:
                    s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (p['color'][0], p['color'][1], p['color'][2], alpha), (size, size), size)
                    self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        # --- Render Maze Walls ---
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                px1 = self.MAZE_AREA_RECT.left + x * self.CELL_WIDTH
                py1 = self.MAZE_AREA_RECT.top + y * self.CELL_HEIGHT
                px2 = px1 + self.CELL_WIDTH
                py2 = py1 + self.CELL_HEIGHT
                
                cell = self.maze_grid[y][x]
                if cell['N']: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px2, py1), 2)
                if cell['S']: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py2), (px2, py2), 2)
                if cell['W']: pygame.draw.line(self.screen, self.COLOR_WALL, (px1, py1), (px1, py2), 2)
                if cell['E']: pygame.draw.line(self.screen, self.COLOR_WALL, (px2, py1), (px2, py2), 2)
        
        # --- Render Keys ---
        key_size = min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.5
        for kx, ky in self.key_locations:
            if [kx, ky] not in self.collected_keys:
                center_x = self.MAZE_AREA_RECT.left + (kx + 0.5) * self.CELL_WIDTH
                center_y = self.MAZE_AREA_RECT.top + (ky + 0.5) * self.CELL_HEIGHT
                rect = pygame.Rect(center_x - key_size / 2, center_y - key_size / 2, key_size, key_size)
                pygame.draw.rect(self.screen, self.COLOR_KEY, rect, 3, border_radius=3)
        
        # --- Render Exit ---
        exit_size = min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.8
        ex, ey = self.exit_pos
        center_x = self.MAZE_AREA_RECT.left + (ex + 0.5) * self.CELL_WIDTH
        center_y = self.MAZE_AREA_RECT.top + (ey + 0.5) * self.CELL_HEIGHT
        rect = pygame.Rect(center_x - exit_size / 2, center_y - exit_size / 2, exit_size, exit_size)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect, border_radius=4)
        
        # --- Render Player ---
        player_size = min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.6
        px, py = self.player_pos
        center_x = self.MAZE_AREA_RECT.left + (px + 0.5) * self.CELL_WIDTH
        center_y = self.MAZE_AREA_RECT.top + (py + 0.5) * self.CELL_HEIGHT
        
        color = self.COLOR_PLAYER
        if self.steps == self.last_action_step: # Pulse on action
            player_size *= 1.2
            color = self.COLOR_PLAYER_PULSE

        rect = pygame.Rect(center_x - player_size / 2, center_y - player_size / 2, player_size, player_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_ui(self):
        # --- Render Moves Remaining ---
        moves_text = self.font_main.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # --- Render Collected Keys ---
        key_ui_size = 20
        for i in range(3):
            rect = pygame.Rect(self.SCREEN_WIDTH - 40 - i * (key_ui_size + 10), 20, key_ui_size, key_ui_size)
            if i < len(self.collected_keys):
                pygame.draw.rect(self.screen, self.COLOR_KEY, rect, border_radius=3)
            else:
                pygame.draw.rect(self.screen, self.COLOR_WALL, rect, 3, border_radius=3)
        
        # --- Render Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_EXIT)
            elif self.remaining_moves <= 0:
                end_text = self.font_large.render("OUT OF MOVES", True, self.COLOR_KEY)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_PLAYER)

            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "collected_keys": len(self.collected_keys),
        }

    def _spawn_particles(self, grid_pos, color):
        px, py = grid_pos
        center_x = self.MAZE_AREA_RECT.left + (px + 0.5) * self.CELL_WIDTH
        center_y = self.MAZE_AREA_RECT.top + (py + 0.5) * self.CELL_HEIGHT
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': self.np_random.integers(3, 7)
            })

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To play manually, you would need a different setup to capture keyboard events
    # and map them to actions. This example just shows the environment running.
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Run a few random steps
    for i in range(200):
        if done:
            print(f"Episode finished after {i+1} steps. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            done = False
            total_reward = 0

        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- To visualize the game (requires a display) ---
        try:
            import os
            if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
                if 'window' not in locals():
                    pygame.display.init()
                    window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
                    pygame.display.set_caption("Maze Runner")
                
                # Pygame expects (W, H) but our obs is (H, W, 3)
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                window.blit(surf, (0, 0))
                pygame.display.flip()
                env.clock.tick(10) # Limit FPS for visualization
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
            if done:
                break
        except (pygame.error, NameError):
            if i == 0:
                print("Running in headless mode. No visualization will be shown.")
            pass

    env.close()
    pygame.quit()