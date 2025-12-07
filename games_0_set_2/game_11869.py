import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:43:44.198055
# Source Brief: brief_01869.md
# Brief Index: 1869
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a bioluminescent kelp forest by matching tiles to create currents. "
        "Push dangerous creatures into traps to descend deeper and escape."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to teleport or select tiles. "
        "Press shift to confirm a match and create a current."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Sizing
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    TILE_SIZE = 40
    GRID_WIDTH, GRID_HEIGHT = GRID_COLS * TILE_SIZE, GRID_ROWS * TILE_SIZE

    # Colors (Bioluminescent Theme)
    COLOR_BG = (5, 10, 25)
    COLOR_GRID_LINE = (20, 40, 80)
    COLOR_PLAYER = (100, 255, 255)
    COLOR_PLAYER_GLOW = (100, 255, 255, 50)
    COLOR_ENEMY = (255, 100, 50)
    COLOR_ENEMY_GLOW = (255, 100, 50, 50)
    COLOR_TRAP = (180, 50, 255)
    COLOR_TRAP_GLOW = (180, 50, 255, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    TILE_COLORS = [
        (0, 150, 255),    # Blue
        (50, 220, 120),    # Green
        (255, 200, 0),      # Yellow
        (255, 100, 200),    # Pink
        (150, 100, 255),    # Purple
    ]

    # Game Parameters
    MAX_STEPS = 2000
    WIN_DEPTH = 20
    MIN_MATCH_COUNT = 3
    PLAYER_MODES = ["TELEPORT", "MATCH"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.reset()
        
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.depth = 1
        self.pending_reward = 0

        # Player state
        self.player_mode = "TELEPORT"
        self.player_pos_grid = None  # (col, row)
        self.cursor_pos_grid = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.selected_tiles = []
        self._update_teleport_range()
        self.teleports_remaining = 2

        # Action state
        self.last_space_held = False
        self.last_shift_held = False

        # Entities and effects
        self.enemies = []
        self.traps = []
        self.particles = []
        self.currents = []
        self.background_kelp = self._generate_background_kelp()
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.pending_reward = 0
        
        turn_taken = self._handle_action(action)

        if turn_taken:
            self._update_game_state()

        terminated = self._check_termination()
        reward = self.pending_reward
        self.score += reward

        # The truncated flag is always False because the episode ends based on game conditions (win/loss/steps)
        # not an artificial time limit imposed by the environment wrapper.
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        turn_taken = False

        # --- Update cursor position ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos_grid = (
                max(0, min(self.GRID_COLS - 1, self.cursor_pos_grid[0] + dx)),
                max(0, min(self.GRID_ROWS - 1, self.cursor_pos_grid[1] + dy))
            )

        # --- Handle mode-specific actions ---
        if self.player_mode == "TELEPORT":
            if space_pressed and self._is_cursor_in_teleport_range():
                # SFX: Teleport_Activate.wav
                self._create_particles(self._grid_to_pixel(self.cursor_pos_grid), 30, self.COLOR_PLAYER)
                self.player_pos_grid = self.cursor_pos_grid
                self.player_mode = "MATCH"
                self.teleports_remaining -= 1
                self.selected_tiles = []
                turn_taken = True

        elif self.player_mode == "MATCH":
            if space_pressed:
                # SFX: Tile_Select.wav
                if self.cursor_pos_grid not in self.selected_tiles:
                    if not self.selected_tiles or (self._is_adjacent(self.cursor_pos_grid, self.selected_tiles[-1]) and self.grid[self.cursor_pos_grid] == self.grid[self.selected_tiles[0]]):
                        self.selected_tiles.append(self.cursor_pos_grid)
                else:
                    # Allow deselecting the last tile
                    if self.selected_tiles and self.cursor_pos_grid == self.selected_tiles[-1]:
                        self.selected_tiles.pop()

            if shift_pressed and len(self.selected_tiles) >= self.MIN_MATCH_COUNT:
                # SFX: Match_Confirm.wav
                self._process_match()
                turn_taken = True
                if not self.game_over:
                    self.player_mode = "TELEPORT"
                    if self.teleports_remaining <= 0:
                        self._progress_to_next_depth()

        return turn_taken

    def _update_game_state(self):
        # Update currents and apply forces
        for current in self.currents:
            for enemy in self.enemies:
                if self._is_in_rect(enemy.pos, current['rect']):
                    enemy.pos = (enemy.pos[0] + current['dir'][0] * 20, enemy.pos[1] + current['dir'][1] * 20)
        self.currents.clear()

        # Update enemies
        for enemy in self.enemies[:]:
            enemy.update(self.grid)
            # Check for collision with player
            if self.player_pos_grid:
                player_px = self._grid_to_pixel(self.player_pos_grid)
                if math.hypot(enemy.pos[0] - player_px[0], enemy.pos[1] - player_px[1]) < self.TILE_SIZE * 0.8:
                    # SFX: Player_Hit.wav
                    self.game_over = True
                    self.pending_reward -= 100
                    return
            # Check for collision with traps
            for trap_pos in self.traps:
                trap_px = self._grid_to_pixel(trap_pos)
                if math.hypot(enemy.pos[0] - trap_px[0], enemy.pos[1] - trap_px[1]) < self.TILE_SIZE * 0.8:
                    # SFX: Enemy_Trapped.wav
                    self._create_particles(trap_px, 50, self.COLOR_TRAP)
                    self.enemies.remove(enemy)
                    self.pending_reward += 5
                    break

        # Update particles
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95)
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _process_match(self):
        match_len = len(self.selected_tiles)
        self.pending_reward += 0.1 if match_len == 3 else 0.5

        # Create current
        start_pos = self.selected_tiles[0]
        end_pos = self.selected_tiles[-1]
        direction = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        norm = math.hypot(*direction)
        if norm > 0:
            direction = (direction[0] / norm, direction[1] / norm)

        # Define current area of effect
        all_x = [pos[0] for pos in self.selected_tiles]
        all_y = [pos[1] for pos in self.selected_tiles]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        rect_px = pygame.Rect(min_x * self.TILE_SIZE, min_y * self.TILE_SIZE, (max_x - min_x + 1) * self.TILE_SIZE, (max_y - min_y + 1) * self.TILE_SIZE)
        
        self.currents.append({'rect': rect_px, 'dir': direction})
        
        # SFX: Current_Flow.wav
        for i in range(match_len * 10):
            tile_idx = i % match_len
            pos = self._grid_to_pixel(self.selected_tiles[tile_idx])
            self.particles.append({
                'pos': (pos[0] + self.np_random.uniform(-10, 10), pos[1] + self.np_random.uniform(-10, 10)),
                'vel': (direction[0] * 3, direction[1] * 3),
                'life': self.np_random.integers(15, 31),
                'color': (255, 255, 100), 'radius': self.np_random.integers(2, 5)
            })

        # Remove matched tiles and let new ones fall
        for pos in sorted(self.selected_tiles, key=lambda p: p[1], reverse=True):
            self._create_particles(self._grid_to_pixel(pos), 15, self.TILE_COLORS[self.grid[pos]])
            col, row = pos
            for r in range(row, 0, -1):
                self.grid[col, r] = self.grid[col, r-1]
            self.grid[col, 0] = self.np_random.integers(0, len(self.TILE_COLORS))
        
        self.selected_tiles = []

    def _progress_to_next_depth(self):
        # SFX: Level_Complete.wav
        self.depth += 1
        self.pending_reward += 10
        if self.depth >= self.WIN_DEPTH:
            self.win = True
            self.game_over = True
            self.pending_reward += 100
        else:
            self._update_teleport_range()
            self.teleports_remaining = 2
            self._generate_level()
            self.player_pos_grid = None
            self.player_mode = "TELEPORT"

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_traps()
        self._render_selection_and_cursor()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "depth": self.depth,
            "player_mode": self.player_mode,
            "teleports_remaining": self.teleports_remaining,
        }

    # --- RENDER METHODS ---
    def _render_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.TILE_COLORS[self.grid[c, r]]
                rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

    def _render_player(self):
        if self.player_pos_grid:
            pos_px = self._grid_to_pixel(self.player_pos_grid)
            self._draw_glow_circle(pos_px, self.TILE_SIZE // 2, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(pos_px[0]), int(pos_px[1]), self.TILE_SIZE // 3, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(pos_px[0]), int(pos_px[1]), self.TILE_SIZE // 3, self.COLOR_PLAYER)

    def _render_enemies(self):
        for enemy in self.enemies:
            self._draw_glow_circle(enemy.pos, self.TILE_SIZE // 2.5, self.COLOR_ENEMY_GLOW)
            points = self._get_triangle_points(enemy.pos, self.TILE_SIZE * 0.6)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENEMY)

    def _render_traps(self):
        for pos_grid in self.traps:
            pos_px = self._grid_to_pixel(pos_grid)
            self._draw_glow_circle(pos_px, self.TILE_SIZE, self.COLOR_TRAP_GLOW)
            # Animated whirlpool
            angle = (pygame.time.get_ticks() / 20) % 360
            for i in range(4):
                radius = self.TILE_SIZE / 2 * (0.2 + 0.8 * (i/4))
                start_angle = math.radians(angle + i * 90)
                end_angle = math.radians(angle + i * 90 + 60)
                rect = pygame.Rect(pos_px[0] - radius, pos_px[1] - radius, radius * 2, radius * 2)
                pygame.draw.arc(self.screen, self.COLOR_TRAP, rect, start_angle, end_angle, 3)

    def _render_selection_and_cursor(self):
        # Draw teleport range
        if self.player_mode == 'TELEPORT':
            if self.player_pos_grid:
                center_px = self._grid_to_pixel(self.player_pos_grid)
                radius_px = self.teleport_range * self.TILE_SIZE
                pygame.gfxdraw.filled_circle(self.screen, int(center_px[0]), int(center_px[1]), int(radius_px), (0, 50, 80, 100))
                pygame.gfxdraw.aacircle(self.screen, int(center_px[0]), int(center_px[1]), int(radius_px), (0, 100, 150, 150))
        
        # Draw selected tiles
        for pos in self.selected_tiles:
            rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

        # Draw cursor
        cursor_color = self.COLOR_CURSOR
        if self.player_mode == 'TELEPORT' and not self._is_cursor_in_teleport_range():
            cursor_color = (128, 128, 128) # Gray out if invalid
        rect = pygame.Rect(self.cursor_pos_grid[0] * self.TILE_SIZE, self.cursor_pos_grid[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, cursor_color, rect, 3)

    def _render_particles(self):
        for p in self.particles:
            self._draw_glow_circle(p['pos'], p['radius'] * 2, (*p['color'], 50))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_background(self):
        for kelp in self.background_kelp:
            points = []
            for i in range(kelp['segments']):
                x = kelp['x'] + math.sin(kelp['freq'] * i + pygame.time.get_ticks() * 0.001 * kelp['speed']) * kelp['amp']
                y = self.SCREEN_HEIGHT - i * (self.SCREEN_HEIGHT / kelp['segments'])
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, kelp['color'], False, points, kelp['width'])

    def _render_ui(self):
        depth_text = self.font_ui.render(f"Depth: {self.depth}/{self.WIN_DEPTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(depth_text, (10, 5))

        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        for i in range(self.teleports_remaining):
            pos = (self.SCREEN_WIDTH - 30 - i * 25, 15)
            self._draw_glow_circle(pos, 12, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY" if self.win else "GAME OVER"
        text = self.font_game_over.render(msg, True, self.COLOR_PLAYER if self.win else self.COLOR_ENEMY)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    # --- HELPER METHODS ---
    def _generate_level(self):
        self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_COLS, self.GRID_ROWS))
        self.traps.clear()
        self.enemies.clear()

        # Add traps
        num_traps = 2 + self.depth // 5
        for _ in range(num_traps):
            pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            self.traps.append(pos)
        
        # Add enemies
        num_enemies = 1 + self.depth // 3
        enemy_speed = 1.0 + (self.depth // 5) * 0.5
        for _ in range(num_enemies):
            start_pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            patrol_points = [start_pos]
            for _ in range(self.np_random.integers(2, 5)):
                patrol_points.append((self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)))
            self.enemies.append(Enemy(patrol_points, enemy_speed))

    def _generate_background_kelp(self):
        kelp_list = []
        for _ in range(15):
            kelp_list.append({
                'x': self.np_random.integers(0, self.SCREEN_WIDTH),
                'segments': self.np_random.integers(10, 31),
                'amp': self.np_random.uniform(2, 8),
                'freq': self.np_random.uniform(0.1, 0.4),
                'speed': self.np_random.uniform(0.5, 1.5),
                'color': (self.np_random.integers(10, 21), self.np_random.integers(30, 61), self.np_random.integers(40, 81)),
                'width': self.np_random.integers(1, 4)
            })
        return kelp_list

    def _update_teleport_range(self):
        if self.depth >= 15: self.teleport_range = 6
        elif self.depth >= 10: self.teleport_range = 5
        elif self.depth >= 5: self.teleport_range = 4
        else: self.teleport_range = 3

    def _is_cursor_in_teleport_range(self):
        if not self.player_pos_grid: return True # First teleport is free
        dist = math.hypot(self.cursor_pos_grid[0] - self.player_pos_grid[0], self.cursor_pos_grid[1] - self.player_pos_grid[1])
        return dist <= self.teleport_range

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _grid_to_pixel(self, grid_pos):
        return (grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2, grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)

    def _pixel_to_grid(self, pixel_pos):
        return (int(pixel_pos[0] // self.TILE_SIZE), int(pixel_pos[1] // self.TILE_SIZE))

    def _is_in_rect(self, pos_px, rect_px):
        return rect_px.collidepoint(pos_px)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 41),
                'color': color, 'radius': self.np_random.integers(2, 6)
            })

    def _draw_glow_circle(self, pos, radius, color):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_triangle_points(self, center, size):
        angle_rad = math.atan2(center[1] - self.SCREEN_HEIGHT/2, center[0] - self.SCREEN_WIDTH/2) # Pointing roughly outwards
        p1 = (center[0] + math.cos(angle_rad) * size / 2, center[1] + math.sin(angle_rad) * size / 2)
        p2 = (center[0] + math.cos(angle_rad + 2.2) * size / 2, center[1] + math.sin(angle_rad + 2.2) * size / 2)
        p3 = (center[0] + math.cos(angle_rad - 2.2) * size / 2, center[1] + math.sin(angle_rad - 2.2) * size / 2)
        return p1, p2, p3

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Enemy:
    def __init__(self, patrol_points, speed):
        self.patrol_points_grid = patrol_points
        self.speed = speed
        self.current_target_idx = 1
        self.pos = self._grid_to_pixel(self.patrol_points_grid[0])

    def update(self, grid):
        target_pos_grid = self.patrol_points_grid[self.current_target_idx]
        target_pos_px = self._grid_to_pixel(target_pos_grid)

        dist_x = target_pos_px[0] - self.pos[0]
        dist_y = target_pos_px[1] - self.pos[1]
        dist = math.hypot(dist_x, dist_y)

        if dist < self.speed:
            self.pos = target_pos_px
            self.current_target_idx = (self.current_target_idx + 1) % len(self.patrol_points_grid)
        else:
            self.pos = (self.pos[0] + (dist_x / dist) * self.speed, self.pos[1] + (dist_y / dist) * self.speed)

    def _grid_to_pixel(self, grid_pos):
        return (grid_pos[0] * GameEnv.TILE_SIZE + GameEnv.TILE_SIZE / 2, grid_pos[1] * GameEnv.TILE_SIZE + GameEnv.TILE_SIZE / 2)


# Example usage:
if __name__ == '__main__':
    # This block will not be run in the testing environment, but is useful for local debugging.
    # To run, you'll need to `pip install pygame`.
    # It sets the video driver to a real one, not the dummy one.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # Arrows: Move cursor
    # Space: Teleport / Select tile
    # Shift: Confirm match
    # Q: Quit
    
    running = True
    display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Kelp Forest Escape")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Create a display surface and show the rendered observation
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for playability

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Final Depth: {info['depth']}")
            pygame.time.wait(3000) # Pause for 3 seconds on game over
            obs, info = env.reset()

    pygame.quit()