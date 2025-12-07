
# Generated: 2025-08-27T12:38:30.780644
# Source Brief: brief_00114.md
# Brief Index: 114

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move the selector. Press Space to clear a group of tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by matching groups of 3 or more same-colored tiles in an isometric grid. "
        "Plan your moves carefully to win before you run out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game mechanics
    GRID_WIDTH = 6
    GRID_HEIGHT = 8
    INITIAL_MOVES = 30
    INITIAL_GREY_TILES = 5
    MIN_GROUP_CLEAR = 3
    MAX_STEPS = 500

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 60, 70)
    TILE_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 100, 220),  # Blue
        (220, 220, 50),  # Yellow
        (160, 50, 220),  # Purple
    ]
    COLOR_GREY = (128, 128, 128)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)

    # Rendering
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_ISO_WIDTH = 52
    TILE_ISO_HEIGHT = 26
    TILE_DEPTH = 12

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = 100

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.win_message = ""

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        
        self._generate_board()
        
        # Ensure the starting board has at least one valid move
        while not self._check_for_any_valid_moves():
            self._shuffle_board()

        return self._get_observation(), self._get_info()
    
    def _generate_board(self):
        self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Place initial grey tiles
        placed_greys = 0
        while placed_greys < self.INITIAL_GREY_TILES:
            x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if self.grid[x, y] != -1: # -1 represents grey
                self.grid[x, y] = -1
                placed_greys += 1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0
        terminated = False
        self.steps += 1

        # 1. Handle cursor movement (no move cost)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle selection (costs a move)
        if space_pressed:
            self.moves_remaining -= 1
            
            cx, cy = self.cursor_pos
            tile_color_idx = self.grid[cx, cy]

            if tile_color_idx == -1 or tile_color_idx == -2: # Grey or Empty
                reward = -0.2
            else:
                connected_tiles = self._find_connected_tiles(cx, cy)
                
                if len(connected_tiles) < self.MIN_GROUP_CLEAR:
                    reward = -0.2
                else:
                    # Successful clear
                    # sound: tile_clear.wav
                    clear_color = self.TILE_COLORS[tile_color_idx]
                    cleared_positions = []
                    
                    # Calculate reward
                    num_cleared = len(connected_tiles)
                    if num_cleared == 3: reward += 1
                    elif num_cleared == 4: reward += 2
                    else: reward += 3
                    if num_cleared > 5: reward += 5

                    # Clear tiles and create particles
                    for x, y in connected_tiles:
                        self.grid[x, y] = -2  # Mark as empty
                        cleared_positions.append(self._world_to_screen(x, y))
                    
                    self._create_particles(cleared_positions, clear_color)
                    self._clear_adjacent_greys(connected_tiles)
                    self._apply_gravity_and_refill()
                    
                    # After refilling, check for softlock
                    if not self._check_for_any_valid_moves() and np.sum(self.grid != -2) > 0:
                        # sound: shuffle.wav
                        self._shuffle_board()
                        # Small penalty for forcing a shuffle
                        reward -= 1

            self.score += reward

        # 3. Check for termination conditions
        # Win condition: board is empty
        if np.sum(self.grid != -2) == 0:
            self.game_over = True
            terminated = True
            if self.moves_remaining >= 10:
                reward += 100
                self.win_message = "PERFECT CLEAR! (+100)"
            else:
                reward += 50
                self.win_message = "Board Cleared! (+50)"
            self.score += reward
            # sound: win_fanfare.wav

        # Loss condition: out of moves
        if self.moves_remaining <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward -= 100
            self.score += reward
            self.win_message = "OUT OF MOVES (-100)"
            # sound: lose_sound.wav
        
        # Max steps termination
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

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid_and_tiles()
        self._draw_cursor()
        self._update_and_render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
        }

    # --- Helper & Rendering Methods ---

    def _world_to_screen(self, x, y):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.grid_offset_x + (x - y) * self.TILE_ISO_WIDTH / 2
        screen_y = self.grid_offset_y + (x + y) * self.TILE_ISO_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _render_iso_tile(self, surface, pos, color, depth_color, side_color):
        """Renders a single isometric tile with a 3D effect."""
        x, y = pos
        # Top face
        points_top = [
            (x, y - self.TILE_ISO_HEIGHT / 2),
            (x + self.TILE_ISO_WIDTH / 2, y),
            (x, y + self.TILE_ISO_HEIGHT / 2),
            (x - self.TILE_ISO_WIDTH / 2, y),
        ]
        pygame.gfxdraw.aapolygon(surface, points_top, color)
        pygame.gfxdraw.filled_polygon(surface, points_top, color)

        # Left face
        points_left = [
            (x - self.TILE_ISO_WIDTH / 2, y),
            (x, y + self.TILE_ISO_HEIGHT / 2),
            (x, y + self.TILE_ISO_HEIGHT / 2 + self.TILE_DEPTH),
            (x - self.TILE_ISO_WIDTH / 2, y + self.TILE_DEPTH)
        ]
        pygame.gfxdraw.aapolygon(surface, points_left, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)

        # Right face
        points_right = [
            (x + self.TILE_ISO_WIDTH / 2, y),
            (x, y + self.TILE_ISO_HEIGHT / 2),
            (x, y + self.TILE_ISO_HEIGHT / 2 + self.TILE_DEPTH),
            (x + self.TILE_ISO_WIDTH / 2, y + self.TILE_DEPTH)
        ]
        pygame.gfxdraw.aapolygon(surface, points_right, depth_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, depth_color)
    
    def _draw_grid_and_tiles(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._world_to_screen(x, y)
                tile_idx = self.grid[x, y]
                
                if tile_idx == -2: # Empty
                    continue
                
                color = self.COLOR_GREY if tile_idx == -1 else self.TILE_COLORS[tile_idx]
                
                # Create darker shades for 3D effect
                depth_color = tuple(max(0, c - 40) for c in color)
                side_color = tuple(max(0, c - 20) for c in color)
                
                self._render_iso_tile(self.screen, screen_pos, color, depth_color, side_color)

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        screen_x, screen_y = self._world_to_screen(cx, cy)
        
        points = [
            (screen_x, screen_y - self.TILE_ISO_HEIGHT / 2 - 2),
            (screen_x + self.TILE_ISO_WIDTH / 2 + 2, screen_y),
            (screen_x, screen_y + self.TILE_ISO_HEIGHT / 2 + 2),
            (screen_x - self.TILE_ISO_WIDTH / 2 - 2, screen_y),
        ]
        
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 3)

    def _render_ui(self):
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 10))

        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _find_connected_tiles(self, start_x, start_y):
        target_color = self.grid[start_x, start_y]
        if target_color < 0: return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        connected = []

        while q:
            x, y = q.popleft()
            connected.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return connected

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == -2: # Empty
                    empty_slots.append(y)
                elif empty_slots:
                    dest_y = empty_slots.pop(0)
                    self.grid[x, dest_y] = self.grid[x, y]
                    self.grid[x, y] = -2
                    empty_slots.append(y)
            
            # Refill top with new tiles
            for y in empty_slots:
                self.grid[x, y] = self.np_random.integers(0, len(self.TILE_COLORS))

    def _clear_adjacent_greys(self, cleared_tiles):
        # sound: grey_break.wav
        greys_to_clear = set()
        for x, y in cleared_tiles:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if self.grid[nx, ny] == -1: # Is a grey tile
                        greys_to_clear.add((nx, ny))
        
        for gx, gy in greys_to_clear:
            self.grid[gx, gy] = -2 # Mark as empty

    def _check_for_any_valid_moves(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if not visited[x, y] and self.grid[x, y] >= 0:
                    connected = self._find_connected_tiles(x, y)
                    if len(connected) >= self.MIN_GROUP_CLEAR:
                        return True
                    for cx, cy in connected:
                        visited[cx, cy] = True
        return False

    def _shuffle_board(self):
        """Shuffles colored tiles, keeping grey tiles and empty spaces fixed."""
        colored_tiles = []
        positions = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] >= 0: # Is a colored tile
                    colored_tiles.append(self.grid[x, y])
                    positions.append((x, y))
        
        self.np_random.shuffle(colored_tiles)
        
        for i, (x, y) in enumerate(positions):
            self.grid[x, y] = colored_tiles[i]

    def _create_particles(self, positions, color):
        for pos in positions:
            for _ in range(10): # 10 particles per cleared tile
                particle = {
                    "pos": list(pos),
                    "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)],
                    "lifespan": self.np_random.integers(20, 40),
                    "color": color,
                    "radius": self.np_random.uniform(2, 5)
                }
                self.particles.append(particle)

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity on particles
            p["lifespan"] -= 1
            
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p["lifespan"] / 20))))
                color = (*p["color"], alpha)
                
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
                self.screen.blit(temp_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Isometric Tile Matcher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop for human play
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # No-op default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q: # Quit
                    running = False

        # Only step if an action was taken
        if np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}. Press 'R' to restart.")
                done = True

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()