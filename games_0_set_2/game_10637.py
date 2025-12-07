import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:44:24.267874
# Source Brief: brief_00637.md
# Brief Index: 637
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a dark labyrinth, avoiding patrolling sentinels. Use shape-based tools to unlock new paths and find your way to the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press 'shift' to cycle tools and 'space' to use a tool on an adjacent locked path."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    TILE_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    TILE_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT
    MAX_STEPS = 1000
    INTERP_SPEED = 0.3  # Speed for visual interpolation

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_BG_ACCENT = (25, 40, 60)
    COLOR_OBSTACLE = (50, 60, 80)
    COLOR_PLAYER = (255, 215, 0) # Gold
    COLOR_PLAYER_GLOW = (255, 215, 0, 50)
    COLOR_ENEMY = (255, 50, 50) # Red
    COLOR_ENEMY_GLOW = (255, 50, 50, 70)
    COLOR_EXIT = (255, 215, 0)
    COLOR_EXIT_GLOW = (255, 215, 0, 100)
    COLOR_PATH_HIDDEN = (70, 90, 120)
    COLOR_PATH_REVEALED = (100, 180, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 45, 65, 180)
    COLOR_PROXIMITY_WARNING = (255, 50, 50)

    # Tile types
    TILE_EMPTY = 0
    TILE_OBSTACLE = 1
    TILE_PATH_REVEALED = 2
    TILE_EXIT = 3
    # Locked paths are encoded as 10 + shape_id
    LOCKED_PATH_OFFSET = 10
    
    # Shapes (for tools and locks)
    SHAPE_CIRCLE = 0
    SHAPE_SQUARE = 1
    SHAPE_TRIANGLE = 2
    NUM_SHAPES = 3

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.grid = None
        self.player_pos = (0, 0)
        self.player_visual_pos = [0.0, 0.0]
        self.exit_pos = (0, 0)
        self.enemies = []
        self.player_inventory = [self.SHAPE_CIRCLE, self.SHAPE_SQUARE, self.SHAPE_TRIANGLE]
        self.selected_tool_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self._generate_map()
        
        self.player_visual_pos = [self.player_pos[0] * self.TILE_WIDTH, self.player_pos[1] * self.TILE_HEIGHT]
        
        self.enemies = []
        self._spawn_enemy() # Initial enemy
        
        self.selected_tool_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Store pre-action state for reward calculation ---
        prev_dist_to_exit = self._distance(self.player_pos, self.exit_pos)
        prev_min_dist_to_enemy = self._get_min_dist_to_enemy()

        # --- Handle Actions (rising edge detection) ---
        # Cycle tool
        if shift_held and not self.last_shift_held:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.player_inventory)
        
        # Use tool
        path_revealed = False
        if space_held and not self.last_space_held:
            path_revealed = self._use_tool()
            if path_revealed:
                reward += 0.1

        # Player Movement
        self._update_player_movement(movement)
        
        # --- Update Game State ---
        self._update_enemies()

        # --- Calculate Rewards ---
        # Distance to exit reward
        new_dist_to_exit = self._distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < prev_dist_to_exit:
            reward += 0.01
        
        # Distance to enemy penalty
        new_min_dist_to_enemy = self._get_min_dist_to_enemy()
        if new_min_dist_to_enemy < prev_min_dist_to_enemy:
            reward -= 0.1

        # --- Check Termination Conditions ---
        terminated = False
        if self._check_win():
            reward += 100.0
            self.score += 100.0
            terminated = True
        
        if self._check_collision():
            reward -= 10.0
            self.score -= 10.0
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
            
        self.game_over = terminated or truncated
        self.score += reward
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    # --- Game Logic Helpers ---
    def _generate_map(self):
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.TILE_OBSTACLE, dtype=int)
        
        # Generate a guaranteed solvable path
        start_y = self.np_random.integers(3, self.GRID_HEIGHT - 3)
        self.player_pos = (1, start_y)
        self.grid[self.player_pos] = self.TILE_EMPTY

        exit_y = self.np_random.integers(3, self.GRID_HEIGHT - 3)
        self.exit_pos = (self.GRID_WIDTH - 2, exit_y)
        
        # Random walk from player to exit to create a path
        path = []
        curr = list(self.player_pos)
        while tuple(curr) != self.exit_pos:
            path.append(tuple(curr))
            # Move towards exit with some randomness
            move_x = np.sign(self.exit_pos[0] - curr[0])
            move_y = np.sign(self.exit_pos[1] - curr[1])
            
            if self.np_random.random() < 0.7: # Prioritize moving towards exit
                if move_x != 0 and curr[0] + move_x < self.GRID_WIDTH -1:
                    curr[0] += move_x
                elif move_y != 0:
                    curr[1] += move_y
            else: # Random move
                if self.np_random.random() < 0.5:
                    curr[0] += self.np_random.choice([-1, 1])
                else:
                    curr[1] += self.np_random.choice([-1, 1])

            curr[0] = np.clip(curr[0], 1, self.GRID_WIDTH - 2)
            curr[1] = np.clip(curr[1], 1, self.GRID_HEIGHT - 2)

        path.append(self.exit_pos)

        # Carve path, placing shape locks
        for i, pos in enumerate(path):
            if i == 0: continue
            if self.grid[pos] == self.TILE_OBSTACLE:
                if i > 1 and i < len(path) - 2 and self.np_random.random() < 0.4:
                    shape_id = self.np_random.integers(0, self.NUM_SHAPES)
                    self.grid[pos] = self.LOCKED_PATH_OFFSET + shape_id
                else:
                    self.grid[pos] = self.TILE_EMPTY
        
        self.grid[self.exit_pos] = self.TILE_EXIT
        self.grid[self.player_pos] = self.TILE_EMPTY

    def _update_player_movement(self, movement):
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right
        
        px = np.clip(px, 0, self.GRID_WIDTH - 1)
        py = np.clip(py, 0, self.GRID_HEIGHT - 1)
        
        tile_type = self.grid[px, py]
        if tile_type in [self.TILE_EMPTY, self.TILE_PATH_REVEALED, self.TILE_EXIT]:
            self.player_pos = (px, py)

    def _use_tool(self):
        tool = self.player_inventory[self.selected_tool_idx]
        px, py = self.player_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                tile_id = self.grid[nx, ny]
                if tile_id >= self.LOCKED_PATH_OFFSET:
                    shape_id = tile_id - self.LOCKED_PATH_OFFSET
                    if shape_id == tool:
                        self.grid[nx, ny] = self.TILE_PATH_REVEALED
                        return True
        return False

    def _update_enemies(self):
        # Spawn new enemies periodically
        if self.steps > 0 and self.steps % 200 == 0:
            self._spawn_enemy()

        # Update existing enemies
        speed_mod = 1.0 - (0.05 * (self.steps // 100))
        move_interval = max(5, int(20 * speed_mod))

        if self.steps % move_interval == 0:
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                path_start, path_end = enemy['path']
                
                # Move along path
                if enemy['dir'] == 1:
                    if ex < path_end[0]: ex += 1
                    elif ey < path_end[1]: ey += 1
                    else: enemy['dir'] = -1
                else:
                    if ex > path_start[0]: ex -= 1
                    elif ey > path_start[1]: ey -= 1
                    else: enemy['dir'] = 1
                
                enemy['pos'] = (ex, ey)

    def _spawn_enemy(self):
        for _ in range(100): # Try to find a valid patrol path
            is_horizontal = self.np_random.random() < 0.5
            if is_horizontal:
                y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
                x1 = self.np_random.integers(1, self.GRID_WIDTH // 2)
                x2 = self.np_random.integers(self.GRID_WIDTH // 2, self.GRID_WIDTH - 2)
                start, end = (x1, y), (x2, y)
            else:
                x = self.np_random.integers(1, self.GRID_WIDTH - 1)
                y1 = self.np_random.integers(1, self.GRID_HEIGHT // 2)
                y2 = self.np_random.integers(self.GRID_HEIGHT // 2, self.GRID_HEIGHT - 2)
                start, end = (x, y1), (x, y2)
            
            # Ensure path is clear of obstacles and not near player start/exit
            path_clear = True
            for i in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                for j in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                     if self.grid[i, j] == self.TILE_OBSTACLE or self._distance((i, j), self.player_pos) < 5 or self._distance((i, j), self.exit_pos) < 5:
                         path_clear = False
                         break
                if not path_clear: break
            
            if path_clear:
                self.enemies.append({'pos': start, 'path': (start, end), 'dir': 1, 'visual_pos': [start[0] * self.TILE_WIDTH, start[1] * self.TILE_HEIGHT]})
                return

    def _check_collision(self):
        return any(self.player_pos == e['pos'] for e in self.enemies)

    def _check_win(self):
        return self.player_pos == self.exit_pos

    def _distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_min_dist_to_enemy(self):
        if not self.enemies:
            return float('inf')
        return min(self._distance(self.player_pos, e['pos']) for e in self.enemies)

    # --- Rendering Helpers ---
    def _interpolate_pos(self, visual_pos, target_grid_pos):
        target_px = [target_grid_pos[0] * self.TILE_WIDTH, target_grid_pos[1] * self.TILE_HEIGHT]
        visual_pos[0] += (target_px[0] - visual_pos[0]) * self.INTERP_SPEED
        visual_pos[1] += (target_px[1] - visual_pos[1]) * self.INTERP_SPEED
        return visual_pos

    def _grid_to_pixel_center(self, grid_pos):
        return (grid_pos[0] * self.TILE_WIDTH + self.TILE_WIDTH // 2,
                grid_pos[1] * self.TILE_HEIGHT + self.TILE_HEIGHT // 2)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(10):
            r = self.COLOR_BG_ACCENT[0] - i
            g = self.COLOR_BG_ACCENT[1] - i
            b = self.COLOR_BG_ACCENT[2] - i
            pygame.draw.rect(self.screen, (r,g,b), (0, i * self.SCREEN_HEIGHT//10, self.SCREEN_WIDTH, self.SCREEN_HEIGHT//10))

    def _render_game_elements(self):
        # Interpolate visual positions
        self.player_visual_pos = self._interpolate_pos(self.player_visual_pos, self.player_pos)
        for enemy in self.enemies:
            enemy['visual_pos'] = self._interpolate_pos(enemy['visual_pos'], enemy['pos'])

        # Render grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                tile_type = self.grid[x, y]
                rect = pygame.Rect(x * self.TILE_WIDTH, y * self.TILE_HEIGHT, self.TILE_WIDTH, self.TILE_HEIGHT)
                center = rect.center
                
                if tile_type == self.TILE_OBSTACLE:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect.inflate(-4, -4))
                elif tile_type == self.TILE_PATH_REVEALED:
                    pygame.draw.rect(self.screen, self.COLOR_PATH_REVEALED, rect)
                elif tile_type >= self.LOCKED_PATH_OFFSET:
                    pygame.draw.rect(self.screen, self.COLOR_PATH_HIDDEN, rect)
                    shape_id = tile_type - self.LOCKED_PATH_OFFSET
                    self._draw_shape(self.screen, shape_id, center, 6, self.COLOR_OBSTACLE, filled=False)

        # Render Exit
        exit_center_px = self._grid_to_pixel_center(self.exit_pos)
        pygame.gfxdraw.filled_circle(self.screen, exit_center_px[0], exit_center_px[1], int(self.TILE_WIDTH * 0.7), self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, exit_center_px[0], exit_center_px[1], int(self.TILE_WIDTH * 0.4), self.COLOR_EXIT)

        # Render Enemies
        for enemy in self.enemies:
            ex_px, ey_px = enemy['visual_pos']
            center = (int(ex_px + self.TILE_WIDTH/2), int(ey_px + self.TILE_HEIGHT/2))
            radius = int(self.TILE_WIDTH * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius + 4, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_ENEMY)

        # Render Player and Proximity Warning
        px_px, py_px = self.player_visual_pos
        player_center = (int(px_px + self.TILE_WIDTH/2), int(py_px + self.TILE_HEIGHT/2))
        player_radius = int(self.TILE_WIDTH * 0.4)
        
        min_dist = self._get_min_dist_to_enemy()
        if min_dist < 4:
            warning_alpha = int(150 * (1 - (min_dist / 4)))
            warning_radius = int(self.TILE_WIDTH * (1 + 2 * (1 - (min_dist / 4))))
            if warning_alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], warning_radius, (*self.COLOR_PROXIMITY_WARNING, warning_alpha))

        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], player_radius + 6, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.SCREEN_HEIGHT - 40))

        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - 30))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (200, self.SCREEN_HEIGHT - 30))

        # Selected Tool
        tool_text = self.font_ui.render("TOOL:", True, self.COLOR_UI_TEXT)
        self.screen.blit(tool_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 30))
        
        tool_pos = (self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 20)
        selected_tool_shape = self.player_inventory[self.selected_tool_idx]
        self._draw_shape(self.screen, selected_tool_shape, tool_pos, 10, self.COLOR_PLAYER)

    def _draw_shape(self, surface, shape_id, pos, size, color, filled=True):
        x, y = int(pos[0]), int(pos[1])
        if shape_id == self.SHAPE_CIRCLE:
            if filled: pygame.gfxdraw.filled_circle(surface, x, y, size, color)
            pygame.gfxdraw.aacircle(surface, x, y, size, color)
        elif shape_id == self.SHAPE_SQUARE:
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            if filled: pygame.draw.rect(surface, color, rect)
            else: pygame.draw.rect(surface, color, rect, 2)
        elif shape_id == self.SHAPE_TRIANGLE:
            points = [
                (x, y - size),
                (x - size, y + size // 2),
                (x + size, y + size // 2)
            ]
            if filled: pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def validate_implementation(self):
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
    # This block allows you to play the game manually for testing
    # For this to work, you must unset the SDL_VIDEODRIVER dummy variable
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Stealth")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()