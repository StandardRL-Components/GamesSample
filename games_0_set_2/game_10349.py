import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Navigate a shifting ziggurat maze by changing size to solve puzzles and find the lost king's tablet.
    - Large size: Push red blocks.
    - Small size: Use blue portals to travel between levels.
    - Goal: Reach the golden tablet on the top level.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Navigate a shifting ziggurat maze by changing size to solve puzzles and find the lost king's tablet."
    user_guide = "Use arrow keys to move. Press Shift to change size and Space to use portals (when small)."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.UI_HEIGHT = 80
        self.GAME_HEIGHT = self.SCREEN_HEIGHT - self.UI_HEIGHT
        self.GRID_SIZE = 32
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.GAME_HEIGHT // self.GRID_SIZE
        self.NUM_LEVELS = 4
        self.MAX_STEPS = 2000
        self.STUCK_LIMIT = 50

        # --- Visuals ---
        self.COLOR_BG = (20, 15, 30)
        self.COLOR_WALL = (60, 50, 80)
        self.COLOR_FLOOR = (40, 30, 60)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_BLOCK = (255, 80, 80)
        self.COLOR_PORTAL = (80, 150, 255)
        self.COLOR_TABLET = (255, 220, 50)
        self.COLOR_UI_BG = (10, 5, 20)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_MINIMAP_CURRENT = (255, 255, 255)
        self.COLOR_MINIMAP_OTHER = (100, 100, 120)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 36)
        except pygame.error:
            self.font_main = pygame.font.SysFont('sans', 24)
            self.font_title = pygame.font.SysFont('sans', 36)
        
        # --- State Variables ---
        self.mazes = []
        self.portals = []
        self.blocks = []
        self.tablet_pos = None
        self.player_pos = None
        self.player_size = 'large'
        self.current_level = 0
        self.steps = 0
        self.score = 0
        self.stuck_counter = 0
        self.visited_tiles = set()
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stuck_counter = 0
        self.current_level = 0
        self.player_size = 'large'
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_ziggurat()

        player_start_tile = self.portals[0]['down']
        self.player_pos = {'x': player_start_tile[0], 'y': player_start_tile[1]}

        self.visited_tiles = set()
        self.visited_tiles.add((self.player_pos['x'], self.player_pos['y'], self.current_level))
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self, width, height):
        grid = np.ones((height, width), dtype=np.uint8)
        
        stack = [(self.np_random.integers(0, width//2)*2, self.np_random.integers(0, height//2)*2)]
        grid[stack[0][1], stack[0][0]] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(0, len(neighbors))]
                grid[ny, nx] = 0
                grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # Make maze more open
        num_holes = (width * height) // (5 + self.np_random.integers(0, 5))
        for _ in range(num_holes):
             rx, ry = self.np_random.integers(1, width-1), self.np_random.integers(1, height-1)
             if grid[ry,rx] == 1:
                grid[ry,rx] = 0

        floor_tiles = []
        for y_idx in range(height):
            for x_idx in range(width):
                if grid[y_idx, x_idx] == 0:
                    floor_tiles.append((x_idx, y_idx))

        return grid, floor_tiles

    def _generate_ziggurat(self):
        self.mazes, self.portals, self.blocks = [], [], []
        all_floor_tiles = []
        for _ in range(self.NUM_LEVELS):
            maze, floor_tiles = self._generate_maze(self.GRID_WIDTH, self.GRID_HEIGHT)
            self.mazes.append(maze)
            all_floor_tiles.append(floor_tiles)

        last_up_portal = None
        for i in range(self.NUM_LEVELS):
            level_portals = {}
            floor = list(all_floor_tiles[i])
            if not floor: floor = [(0,0), (1,1)]

            if i > 0:
                level_portals['down'] = last_up_portal
                if last_up_portal in floor: floor.remove(last_up_portal)
            else:
                start_pos_idx = self.np_random.integers(0, len(floor))
                start_pos = floor[start_pos_idx]
                level_portals['down'] = start_pos
                if start_pos in floor: floor.remove(start_pos)

            if i < self.NUM_LEVELS - 1:
                up_pos_idx = self.np_random.integers(0, len(floor)) if floor else 0
                up_pos = floor[up_pos_idx] if floor else (self.GRID_WIDTH-1, self.GRID_HEIGHT-1)
                level_portals['up'] = up_pos
                if up_pos in floor: floor.remove(up_pos)
                last_up_portal = up_pos
            self.portals.append(level_portals)

        top_floor = all_floor_tiles[self.NUM_LEVELS - 1]
        if not top_floor: top_floor.append((self.GRID_WIDTH-1, self.GRID_HEIGHT-1))
        tablet_tile_idx = self.np_random.integers(0, len(top_floor))
        tablet_tile = top_floor[tablet_tile_idx]
        self.tablet_pos = {'x': tablet_tile[0], 'y': tablet_tile[1], 'level': self.NUM_LEVELS - 1}
        if tablet_tile in top_floor: top_floor.remove(tablet_tile)

        for i in range(1, self.NUM_LEVELS):
            num_blocks = i 
            level_floor = all_floor_tiles[i]
            for _ in range(num_blocks):
                if level_floor:
                    block_pos_idx = self.np_random.integers(0, len(level_floor))
                    block_pos = level_floor[block_pos_idx]
                    self.blocks.append({'x': block_pos[0], 'y': block_pos[1], 'level': i})
                    level_floor.remove(block_pos)
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Action Handling ---
        action_meaningful = False

        # 1. Size Change (on press)
        if shift_held and not self.prev_shift_held:
            self.stuck_counter = 0
            action_meaningful = True
            original_size = self.player_size
            self.player_size = 'small' if self.player_size == 'large' else 'large'
            if self.player_size == 'large':
                px, py = self.player_pos['x'], self.player_pos['y']
                is_stuck = self.mazes[self.current_level][py, px] == 1 or \
                           any(b['x'] == px and b['y'] == py and b['level'] == self.current_level for b in self.blocks)
                if is_stuck: self.player_size = original_size

        # 2. Teleport (on press)
        if space_held and not self.prev_space_held:
            px, py = self.player_pos['x'], self.player_pos['y']
            if self.player_size == 'small':
                level_portals = self.portals[self.current_level]
                if 'up' in level_portals and (px, py) == level_portals['up']:
                    self.stuck_counter = 0
                    action_meaningful = True
                    self.current_level += 1
                    new_pos = self.portals[self.current_level]['down']
                    self.player_pos = {'x': new_pos[0], 'y': new_pos[1]}
                    reward += 1.0; self.score += 1
                    self.visited_tiles.clear()
                elif 'down' in level_portals and (px, py) == level_portals['down'] and self.current_level > 0:
                    self.stuck_counter = 0
                    action_meaningful = True
                    self.current_level -= 1
                    new_pos = self.portals[self.current_level]['up']
                    self.player_pos = {'x': new_pos[0], 'y': new_pos[1]}
                    reward += 1.0; self.score += 1
                    self.visited_tiles.clear()

        # 3. Movement
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
        if dx != 0 or dy != 0:
            tx, ty = self.player_pos['x'] + dx, self.player_pos['y'] + dy
            if 0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT and self.mazes[self.current_level][ty, tx] == 0:
                block = next((b for b in self.blocks if b['level'] == self.current_level and b['x'] == tx and b['y'] == ty), None)
                if block:
                    if self.player_size == 'large':
                        ptx, pty = tx + dx, ty + dy
                        if 0 <= ptx < self.GRID_WIDTH and 0 <= pty < self.GRID_HEIGHT and self.mazes[self.current_level][pty, ptx] == 0:
                            if not any(b['level'] == self.current_level and b['x'] == ptx and b['y'] == pty for b in self.blocks):
                                block['x'], block['y'] = ptx, pty
                                self.player_pos['x'], self.player_pos['y'] = tx, ty
                                reward += 5.0; self.score += 5
                else:
                    self.player_pos['x'], self.player_pos['y'] = tx, ty
            action_meaningful = True # Any move attempt is meaningful
            self.stuck_counter = 0

        # --- State, Reward, and Termination ---
        current_tile = (self.player_pos['x'], self.player_pos['y'], self.current_level)
        if current_tile not in self.visited_tiles:
            self.visited_tiles.add(current_tile)
            reward += 0.1

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        self.steps += 1
        if not action_meaningful and movement == 0: self.stuck_counter += 1

        terminated = False
        truncated = False
        if self.current_level == self.tablet_pos['level'] and self.player_pos['x'] == self.tablet_pos['x'] and self.player_pos['y'] == self.tablet_pos['y']:
            reward += 100.0; self.score += 100
            terminated = True
        elif self.stuck_counter >= self.STUCK_LIMIT:
            reward -= 100.0; self.score -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level}

    def _render_game(self):
        maze = self.mazes[self.current_level]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.COLOR_WALL if maze[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))

        for portal_type, pos in self.portals[self.current_level].items():
            px, py = pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            radius = int(self.GRID_SIZE * 0.3 + pulse * 3)
            for i in range(radius, 0, -2):
                alpha = int(100 * (1 - i / radius)**2)
                pygame.gfxdraw.filled_circle(self.screen, px, py, i, self.COLOR_PORTAL + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PORTAL)

        for block in self.blocks:
            if block['level'] == self.current_level:
                r = pygame.Rect(block['x'] * self.GRID_SIZE + 2, block['y'] * self.GRID_SIZE + 2, self.GRID_SIZE - 4, self.GRID_SIZE - 4)
                pygame.draw.rect(self.screen, self.COLOR_BLOCK, r, border_radius=3)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_BLOCK), r, 2, border_radius=3)

        if self.current_level == self.tablet_pos['level']:
            tx, ty = self.tablet_pos['x'] * self.GRID_SIZE, self.tablet_pos['y'] * self.GRID_SIZE
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            glow_size = int(self.GRID_SIZE * 0.8 + pulse * 5)
            glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_TABLET + (30,), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (tx + self.GRID_SIZE//2 - glow_size, ty + self.GRID_SIZE//2 - glow_size))
            pygame.draw.rect(self.screen, self.COLOR_TABLET, (tx + 8, ty + 8, self.GRID_SIZE - 16, self.GRID_SIZE - 16), border_radius=2)

        px, py = self.player_pos['x'] * self.GRID_SIZE, self.player_pos['y'] * self.GRID_SIZE
        size = self.GRID_SIZE - 4 if self.player_size == 'large' else self.GRID_SIZE // 2
        offset = 2 if self.player_size == 'large' else self.GRID_SIZE // 4
        player_rect = pygame.Rect(px + offset, py + offset, size, size)
        center_x, center_y = player_rect.center
        glow_radius = int(size * 0.9)
        for i in range(glow_radius, 0, -3):
             alpha = int(80 * (1 - i / glow_radius)**2)
             pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, i, self.COLOR_PLAYER + (alpha,))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, self.GAME_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (0, self.GAME_HEIGHT), (self.SCREEN_WIDTH, self.GAME_HEIGHT), 2)
        
        self.screen.blit(self.font_title.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT), (20, self.GAME_HEIGHT + 10))
        self.screen.blit(self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT), (20, self.GAME_HEIGHT + 50))

        self.screen.blit(self.font_main.render("SIZE:", True, self.COLOR_UI_TEXT), (200, self.GAME_HEIGHT + 25))
        if self.player_size == 'large':
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (260, self.GAME_HEIGHT + 20, 30, 30), border_radius=3)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (260 + 7, self.GAME_HEIGHT + 20 + 7, 16, 16), border_radius=3)

        map_x, map_y = 400, self.GAME_HEIGHT + 15
        map_w, map_h = 200, 50
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (map_x, map_y, map_w, map_h), border_radius=3)
        level_w = map_w / self.NUM_LEVELS
        for i in range(self.NUM_LEVELS):
            color = self.COLOR_MINIMAP_CURRENT if i == self.current_level else self.COLOR_MINIMAP_OTHER
            level_rect = pygame.Rect(map_x + i * level_w + 2, map_y + 2, level_w - 4, map_h - 4)
            pygame.draw.rect(self.screen, color, level_rect, 2, border_radius=3)
            if i == self.current_level:
                dot_x = int(level_rect.x + (self.player_pos['x'] / self.GRID_WIDTH) * level_rect.width)
                dot_y = int(level_rect.y + (self.player_pos['y'] / self.GRID_HEIGHT) * level_rect.height)
                pygame.draw.circle(self.screen, self.COLOR_PLAYER, (dot_x, dot_y), 3)
            if i == self.tablet_pos['level']:
                dot_x = int(level_rect.x + (self.tablet_pos['x'] / self.GRID_WIDTH) * level_rect.width)
                dot_y = int(level_rect.y + (self.tablet_pos['y'] / self.GRID_HEIGHT) * level_rect.height)
                pygame.draw.circle(self.screen, self.COLOR_TABLET, (dot_x, dot_y), 3)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The following code is for interactive testing and will not work in a purely
    # headless environment, but is useful for local development.
    try:
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Ziggurat Maze")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        terminated = False

        print("\n--- Controls ---")
        print("Arrows: Move")
        print("Space: Teleport (when small and on a portal)")
        print("Shift: Change Size")
        print("R: Reset Environment")
        print("----------------\n")

        while running:
            keys = pygame.key.get_pressed()
            
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False

            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30)
    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("Running a short headless test instead.")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Headless test finished. Final Score: {info.get('score', 0)}, Total Reward: {total_reward:.2f}, Steps: {info.get('steps', 0)}")


    env.close()