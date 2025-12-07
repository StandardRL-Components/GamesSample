import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to interact with keys and stairs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a cursed clock tower by finding the key on each floor and ascending the stairs. "
        "Avoid deadly traps and escape before the 2-minute timer runs out!"
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Game parameters
    MAX_TIME = 120  # seconds
    MAX_STEPS = MAX_TIME * FPS
    TOTAL_FLOORS = 5
    GRID_WIDTH = 11
    GRID_HEIGHT = 11
    TILE_WIDTH_ISO = 48
    TILE_HEIGHT_ISO = 24

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_FLOOR = (52, 73, 94)
    COLOR_WALL = (44, 62, 80)
    COLOR_TRAP = (231, 76, 60)
    COLOR_KEY = (241, 196, 15)
    COLOR_STAIRS = (155, 89, 182)
    COLOR_PLAYER = (52, 200, 235)
    COLOR_PLAYER_GLOW = (52, 200, 235, 60)
    COLOR_TEXT = (236, 240, 241)
    COLOR_UI_BG = (44, 62, 80, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_floor = pygame.font.SysFont("Consolas", 24, bold=True)

        # World origin for isometric projection
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_ISO) // 3

        # Initialize state variables
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_grid_pos = (0, 0)
        self.player_target_grid_pos = (0, 0)
        self.player_is_moving = False
        self.player_move_progress = 0.0
        self.player_speed = 4.0 / self.FPS  # Tiles per second

        self.current_floor_index = 0
        self.floors = []
        self.timer = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_reward = 0
        self.termination_reason = ""

        self.particles = []

        self.np_random = None

        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # Not needed in final code

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.origin_y + (grid_x + grid_y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _generate_floor(self, floor_level):
        grid = [['wall'] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]

        # Use numpy's random generator
        start_x = self.np_random.integers(1, self.GRID_WIDTH - 1)
        start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)

        # Randomized DFS for maze generation
        stack = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        path_cells = []

        while stack:
            cx, cy = stack[-1]
            grid[cy][cx] = 'floor'
            path_cells.append((cx, cy))

            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < self.GRID_WIDTH - 1 and 1 <= ny < self.GRID_HEIGHT - 1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                grid[ny][nx] = 'floor'
                grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 'floor'
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Place entry/exit
        entry_pos = path_cells[0]
        exit_pos = path_cells[-1]

        # Place key
        key_candidates = [c for c in path_cells if c not in [entry_pos, exit_pos]]
        # np.random.choice on a list of tuples returns a numpy array.
        # Convert it to a tuple to prevent comparison errors later.
        key_pos = tuple(self.np_random.choice(key_candidates))

        # Place traps
        num_traps = min(len(path_cells) - 3, 1 + floor_level)
        # Now that key_pos is a tuple, this comparison is safe.
        trap_candidates = [c for c in path_cells if c not in [entry_pos, exit_pos, key_pos]]
        trap_indices = self.np_random.choice(len(trap_candidates), size=num_traps, replace=False)
        traps = [trap_candidates[i] for i in trap_indices]

        return {
            "grid": grid,
            "entry": entry_pos,
            "exit": exit_pos,
            "key_pos": key_pos,
            "has_key": False,
            "traps": [tuple(t) for t in traps]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.current_floor_index = 0
        self.last_space_held = False
        self.termination_reason = ""
        self.particles.clear()

        self.floors = [self._generate_floor(i) for i in range(self.TOTAL_FLOORS)]

        start_pos = self.floors[0]['entry']
        self.player_grid_pos = start_pos
        self.player_target_grid_pos = start_pos
        self.player_pos = pygame.math.Vector2(self._iso_to_screen(*start_pos))
        self.player_is_moving = False
        self.player_move_progress = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        # --- Handle movement and actions ---
        self._update_player_movement()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Handle movement input
        if not self.player_is_moving:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            if dx != 0 or dy != 0:
                next_x, next_y = self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy
                current_floor = self.floors[self.current_floor_index]
                if 0 <= next_x < self.GRID_WIDTH and 0 <= next_y < self.GRID_HEIGHT and \
                   current_floor['grid'][next_y][next_x] == 'floor':
                    self.player_target_grid_pos = (next_x, next_y)
                    self.player_is_moving = True
                    self.player_move_progress = 0.0

        # Handle interaction (spacebar press)
        if space_held and not self.last_space_held:
            reward += self._handle_interaction()
        self.last_space_held = space_held

        # --- Update game state ---
        self.steps += 1
        self.timer -= 1.0 / self.FPS

        # Update particles
        self._update_particles()

        # --- Check for terminal conditions ---
        terminated = False
        truncated = False
        current_floor = self.floors[self.current_floor_index]

        # 1. Fell in a trap
        if self.player_grid_pos in current_floor['traps']:
            reward = -100
            terminated = True
            self.game_over = True
            self.termination_reason = "You fell into a trap!"
            self._create_explosion(self.player_pos, self.COLOR_TRAP, 50)

        # 2. Ran out of time
        if self.timer <= 0:
            reward = -50
            terminated = True
            self.game_over = True
            self.timer = 0
            self.termination_reason = "Time's up!"

        # 3. Won the game
        if self.current_floor_index >= self.TOTAL_FLOORS:
            reward = 100
            terminated = True
            self.game_over = True
            self.termination_reason = "You escaped!"

        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True # Use truncated for step limit
            self.game_over = True
            self.termination_reason = "Step limit reached."

        self.last_reward = reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_movement(self):
        if self.player_is_moving:
            self.player_move_progress += self.player_speed
            start_screen_pos = pygame.math.Vector2(self._iso_to_screen(*self.player_grid_pos))
            target_screen_pos = pygame.math.Vector2(self._iso_to_screen(*self.player_target_grid_pos))

            if self.player_move_progress >= 1.0:
                self.player_pos = target_screen_pos
                self.player_grid_pos = self.player_target_grid_pos
                self.player_is_moving = False
                self.player_move_progress = 0
            else:
                self.player_pos = start_screen_pos.lerp(target_screen_pos, self.player_move_progress)

    def _handle_interaction(self):
        current_floor = self.floors[self.current_floor_index]

        # Interact with key
        if self.player_grid_pos == current_floor['key_pos'] and not current_floor['has_key']:
            current_floor['has_key'] = True
            current_floor['key_pos'] = None  # Key is collected
            self.score += 10
            self._create_explosion(self.player_pos, self.COLOR_KEY, 20)
            return 1.0

        # Interact with stairs
        if self.player_grid_pos == current_floor['exit'] and current_floor['has_key']:
            self.current_floor_index += 1
            self.score += 50
            if self.current_floor_index < self.TOTAL_FLOORS:
                next_floor = self.floors[self.current_floor_index]
                self.player_grid_pos = next_floor['entry']
                self.player_target_grid_pos = self.player_grid_pos
                self.player_pos = pygame.math.Vector2(self._iso_to_screen(*self.player_grid_pos))
                self._create_explosion(self.player_pos, self.COLOR_STAIRS, 40)
            return 5.0

        return 0.0

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
            "timer": self.timer,
            "floor": self.current_floor_index + 1
        }

    def render(self):
        return self._get_observation()

    def _render_game(self):
        if self.game_over and self.current_floor_index >= self.TOTAL_FLOORS:
             # Don't clear screen on win, show final state
             pass
        elif self.game_over:
            return

        current_floor = self.floors[self.current_floor_index]
        grid = current_floor['grid']

        # Render floor and objects
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_type = grid[y][x]
                screen_x, screen_y = self._iso_to_screen(x, y)

                # Draw floor tiles
                if tile_type == 'floor':
                    points = [
                        (screen_x, screen_y),
                        (screen_x + self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2),
                        (screen_x, screen_y + self.TILE_HEIGHT_ISO),
                        (screen_x - self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_WALL)

                # Draw objects on this tile
                pos = (x, y)
                if pos == current_floor['key_pos']:
                    self._draw_diamond(screen_x, screen_y, self.COLOR_KEY, 10)
                if pos in current_floor['traps']:
                    self._draw_diamond(screen_x, screen_y, self.COLOR_TRAP, 12, filled=False)
                if pos == current_floor['exit']:
                    color = self.COLOR_STAIRS if current_floor['has_key'] else (100, 100, 100)
                    self._draw_diamond(screen_x, screen_y, color, 14, filled=True)

        # Render particles
        self._render_particles()

        # Render player
        if not (self.game_over and self.termination_reason == "You fell into a trap!"):
            self._render_player()

    def _draw_diamond(self, sx, sy, color, size, filled=True):
        points = [
            (sx, sy + self.TILE_HEIGHT_ISO / 2 - size / 2),
            (sx + size / 2, sy + self.TILE_HEIGHT_ISO / 2),
            (sx, sy + self.TILE_HEIGHT_ISO / 2 + size / 2),
            (sx - size / 2, sy + self.TILE_HEIGHT_ISO / 2)
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_player(self):
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        player_y_offset = self.TILE_HEIGHT_ISO / 2

        # Glow effect
        glow_radius = 15
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (px - glow_radius, py - glow_radius + int(player_y_offset)))

        # Player circle
        pygame.gfxdraw.filled_circle(self.screen, px, py + int(player_y_offset), 7, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py + int(player_y_offset), 7, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        mins, secs = divmod(int(self.timer), 60)
        timer_text = self.font_ui.render(f"Time: {mins:02d}:{secs:02d}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Floor number
        if not self.game_over:
            floor_str = f"Floor {self.current_floor_index + 1}" if self.current_floor_index < self.TOTAL_FLOORS else "Escaped!"
            floor_text = self.font_floor.render(floor_str, True, self.COLOR_TEXT)
            self.screen.blit(floor_text, (self.SCREEN_WIDTH // 2 - floor_text.get_width() // 2, self.SCREEN_HEIGHT - 40))

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, 100), pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)

            end_text = self.font_floor.render(self.termination_reason, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, 50))
            s.blit(end_text, text_rect)

            self.screen.blit(s, (0, self.SCREEN_HEIGHT // 2 - 50))

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': random.uniform(10, 25),
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            color = (*p['color'], max(0, min(255, alpha)))
            pos = (int(p['pos'].x), int(p['pos'].y) + self.TILE_HEIGHT_ISO // 2)
            pygame.gfxdraw.filled_circle(self.screen, *pos, int(p['size']), color)

    def close(self):
        pygame.quit()