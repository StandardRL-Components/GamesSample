import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:44:16.092082
# Source Brief: brief_00660.md
# Brief Index: 660
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your hexagonal wall from waves of incoming enemies. Place and move your defensive units to protect the tiles and survive until the final wave."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press Space to pick up a unit and release to place it. Press Shift to clone a new unit when the cooldown is ready."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000 # Increased to allow for 10 waves
        self.WIN_WAVE_COUNT = 10

        # Visual Constants
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 65)
        self.COLOR_TILE = (0, 100, 150)
        self.COLOR_TILE_DAMAGED = (200, 200, 0)
        self.COLOR_PLAYER_UNIT = (0, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_GLOW = (255, 255, 255, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR_BG = (50, 60, 80)
        self.COLOR_UI_HEALTH_BAR = (0, 200, 100)
        self.COLOR_UI_CLONE_BAR = (100, 150, 255)

        # Hex Grid Constants
        self.HEX_RADIUS = 18
        self.HEX_GRID_COLS = 16
        self.HEX_GRID_ROWS = 9
        self.HEX_OFFSET_X = 60
        self.HEX_OFFSET_Y = 55
        
        # Gameplay Constants
        self.INITIAL_PLAYER_UNITS = 3
        self.TILE_MAX_HEALTH = 100
        self.PLAYER_UNIT_MAX_HEALTH = 50
        self.CLONE_COOLDOWN_STEPS = 150 # 5 seconds at 30 FPS
        self.WAVE_CLEAR_DELAY = 90 # 3 seconds

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tiles = {}
        self.player_units = {}
        self.enemies = []
        self.particles = []
        self.cursor_pos = (0,0)
        self.held_unit_key = None
        self.clone_cooldown = 0
        self.current_wave = 0
        self.wave_clear_timer = 0
        self.total_max_tile_health = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._init_tiles()
        self._init_player_state()
        
        self.enemies.clear()
        self.particles.clear()
        
        self.current_wave = 0
        self.wave_clear_timer = self.WAVE_CLEAR_DELAY
        self.clone_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1
        
        # Handle input and get action-based rewards
        action_reward = self._handle_input(action)
        reward += action_reward
        
        # Update game state and get event-based rewards
        event_reward = self._update_game_state()
        reward += event_reward
        
        # Check for termination and get terminal rewards
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 10
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], self.cursor_pos[1] - cursor_speed)
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], self.cursor_pos[1] + cursor_speed)
        elif movement == 3: self.cursor_pos = (self.cursor_pos[0] - cursor_speed, self.cursor_pos[1])
        elif movement == 4: self.cursor_pos = (self.cursor_pos[0] + cursor_speed, self.cursor_pos[1])
        self.cursor_pos = (
            np.clip(self.cursor_pos[0], 0, self.WIDTH - 1),
            np.clip(self.cursor_pos[1], 0, self.HEIGHT - 1)
        )
        
        cursor_grid_pos = self._pixel_to_axial(self.cursor_pos[0], self.cursor_pos[1])

        # --- Action Handling (Detect press events) ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        space_release = not space_held and self.prev_space_held

        # Action: Clone Unit (Shift Press)
        if shift_press and self.clone_cooldown <= 0 and cursor_grid_pos in self.tiles and cursor_grid_pos not in self.player_units:
            self.player_units[cursor_grid_pos] = {'health': self.PLAYER_UNIT_MAX_HEALTH, 'original_pos': cursor_grid_pos}
            self.clone_cooldown = self.CLONE_COOLDOWN_STEPS
            self._create_particles(self._axial_to_pixel(*cursor_grid_pos), self.COLOR_PLAYER_UNIT, 20)
            # sfx: clone_success

        # Action: Pick up Unit (Space Press)
        if space_press and self.held_unit_key is None and cursor_grid_pos in self.player_units:
            self.held_unit_key = cursor_grid_pos
            # sfx: pickup_unit

        # Action: Drop Unit (Space Release)
        if space_release and self.held_unit_key is not None:
            # Check if target is valid (on grid, not occupied by another unit)
            if cursor_grid_pos in self.tiles and (cursor_grid_pos not in self.player_units or cursor_grid_pos == self.held_unit_key):
                unit_data = self.player_units.pop(self.held_unit_key)
                unit_data['original_pos'] = cursor_grid_pos
                self.player_units[cursor_grid_pos] = unit_data
                self._create_particles(self._axial_to_pixel(*cursor_grid_pos), self.COLOR_PLAYER_UNIT, 15, 'burst')
                reward += 1.0 # Teleport reward
                # sfx: teleport_success
            self.held_unit_key = None

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        return reward

    def _update_game_state(self):
        reward = 0
        if self.clone_cooldown > 0: self.clone_cooldown -= 1
        
        # Wave Management
        if not self.enemies and not self.game_over:
            if self.wave_clear_timer > 0:
                self.wave_clear_timer -= 1
            else:
                self.current_wave += 1
                if self.current_wave > 1 and self.current_wave <= self.WIN_WAVE_COUNT:
                    reward += 5.0 # Wave survived reward
                if self.current_wave <= self.WIN_WAVE_COUNT:
                    self._start_new_wave()
                    self.wave_clear_timer = self.WAVE_CLEAR_DELAY

        # Update Enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            enemy['pos'] = (enemy['pos'][0] + enemy['vel'][0], enemy['pos'][1] + enemy['vel'][1])
            grid_pos = self._pixel_to_axial(enemy['pos'][0], enemy['pos'][1])

            if grid_pos in self.tiles:
                # Enemy hit the wall
                enemies_to_remove.append(enemy)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 25, 'impact')
                # sfx: enemy_impact

                if grid_pos in self.player_units:
                    # A unit defends the tile
                    self.player_units[grid_pos]['health'] -= enemy['damage']
                    if self.player_units[grid_pos]['health'] <= 0:
                        del self.player_units[grid_pos]
                        # sfx: unit_destroyed
                else:
                    # Tile takes damage
                    self.tiles[grid_pos]['health'] -= enemy['damage']
                    self.tiles[grid_pos]['damage_timer'] = 10 # frames to flash
                    reward -= 0.5 # Tile damaged penalty
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

        # Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95)
            p['life'] -= 1
            p['radius'] *= 0.96

        # Update Tile Damage Flash
        for tile in self.tiles.values():
            if tile['damage_timer'] > 0:
                tile['damage_timer'] -= 1

        return reward

    def _check_termination(self):
        wall_integrity = self._calculate_wall_integrity()
        
        if wall_integrity <= 0:
            self.game_over = True
            return True, -100.0 # Lose penalty
        
        if self.current_wave > self.WIN_WAVE_COUNT:
            self.game_over = True
            return True, 100.0 # Win reward

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, 0.0 # Time limit
        
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw tiles
        for q, r in self.tiles:
            tile = self.tiles[(q, r)]
            center = self._axial_to_pixel(q, r)
            
            health_ratio = max(0, tile['health'] / self.TILE_MAX_HEALTH)
            color = self.COLOR_TILE
            if tile['damage_timer'] > 0:
                # Flash yellow when damaged
                color = self.COLOR_TILE_DAMAGED
            
            self._draw_hexagon(self.screen, color, center, self.HEX_RADIUS)
            
            # Draw health overlay on damaged tiles
            if health_ratio < 1.0 and tile['damage_timer'] == 0:
                 health_color = (
                     self.COLOR_TILE[0] + (self.COLOR_TILE_DAMAGED[0] - self.COLOR_TILE[0]) * (1 - health_ratio),
                     self.COLOR_TILE[1] + (self.COLOR_TILE_DAMAGED[1] - self.COLOR_TILE[1]) * (1 - health_ratio),
                     self.COLOR_TILE[2] + (self.COLOR_TILE_DAMAGED[2] - self.COLOR_TILE[2]) * (1 - health_ratio)
                 )
                 self._draw_hexagon(self.screen, health_color, center, self.HEX_RADIUS * health_ratio)

            self._draw_hexagon_outline(self.screen, self.COLOR_GRID, center, self.HEX_RADIUS, 1)

        # Draw player units
        for grid_pos, unit in self.player_units.items():
            if grid_pos != self.held_unit_key:
                center = self._axial_to_pixel(*grid_pos)
                pygame.draw.circle(self.screen, self.COLOR_PLAYER_UNIT, center, int(self.HEX_RADIUS * 0.6))
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (enemy['pos'][0]-5, enemy['pos'][1]-5, 10, 10))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            self._draw_glowing_circle(self.screen, p['color'], p['pos'], p['radius'], alpha)
            
        # Draw held unit and cursor
        cursor_grid_pos = self._pixel_to_axial(*self.cursor_pos)
        if cursor_grid_pos in self.tiles:
            cursor_hex_center = self._axial_to_pixel(*cursor_grid_pos)
            self._draw_hexagon_outline(self.screen, self.COLOR_CURSOR, cursor_hex_center, self.HEX_RADIUS, 2)
        else:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, 5, 1)

        if self.held_unit_key is not None:
             pygame.draw.circle(self.screen, self.COLOR_PLAYER_UNIT, self.cursor_pos, int(self.HEX_RADIUS * 0.6))
             pygame.draw.circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, int(self.HEX_RADIUS * 0.7), 2)

    def _render_ui(self):
        # Wall Integrity
        integrity = self._calculate_wall_integrity()
        integrity_text = self.font_small.render(f"WALL INTEGRITY: {integrity:.0%}", True, self.COLOR_UI_TEXT)
        self.screen.blit(integrity_text, (10, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (170, 10, 150, 16))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BAR, (170, 10, 150 * integrity, 16))

        # Wave Number
        wave_str = f"WAVE: {self.current_wave}/{self.WIN_WAVE_COUNT}" if self.current_wave <= self.WIN_WAVE_COUNT else "VICTORY!"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Clone Cooldown
        bar_width = self.WIDTH - 20
        bar_y = self.HEIGHT - 25
        clone_ready = self.clone_cooldown <= 0
        clone_progress = 1.0 - (self.clone_cooldown / self.CLONE_COOLDOWN_STEPS) if not clone_ready else 1.0
        
        clone_text_str = "CLONE READY [SHIFT]" if clone_ready else "RECHARGING..."
        clone_text = self.font_small.render(clone_text_str, True, self.COLOR_UI_TEXT)
        text_pos = (self.WIDTH / 2 - clone_text.get_width() / 2, bar_y)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, bar_y, bar_width, 16))
        if clone_progress > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_CLONE_BAR, (10, bar_y, bar_width * clone_progress, 16))
        self.screen.blit(clone_text, text_pos)

        if self.game_over:
            status = "VICTORY" if self.current_wave > self.WIN_WAVE_COUNT else "GAME OVER"
            status_text = self.font_large.render(status, True, self.COLOR_UI_TEXT)
            pos = (self.WIDTH/2 - status_text.get_width()/2, self.HEIGHT/2 - status_text.get_height()/2)
            self.screen.blit(status_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "wall_integrity": self._calculate_wall_integrity(),
            "player_units": len(self.player_units),
        }

    def _init_tiles(self):
        self.tiles.clear()
        self.total_max_tile_health = 0
        for r in range(self.HEX_GRID_ROWS):
            for q in range(self.HEX_GRID_COLS):
                if (q + r) % 2 == 0: # Create a checkerboard-like pattern for visual interest
                    self.tiles[(q, r)] = {'health': self.TILE_MAX_HEALTH, 'max_health': self.TILE_MAX_HEALTH, 'damage_timer': 0}
                    self.total_max_tile_health += self.TILE_MAX_HEALTH
    
    def _init_player_state(self):
        self.cursor_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.held_unit_key = None
        self.prev_space_held = False
        self.prev_shift_held = False

        self.player_units.clear()
        available_tiles = list(self.tiles.keys())
        # Use self.np_random for reproducibility
        self.np_random.shuffle(available_tiles)
        for i in range(min(self.INITIAL_PLAYER_UNITS, len(available_tiles))):
            pos = available_tiles[i]
            self.player_units[pos] = {'health': self.PLAYER_UNIT_MAX_HEALTH, 'original_pos': pos}

    def _start_new_wave(self):
        self.enemies.clear()
        enemy_count = 3 + (self.current_wave - 1)
        enemy_speed = 1.0 + (self.current_wave - 1) * 0.1
        enemy_damage = 25 + self.current_wave * 5

        for _ in range(enemy_count):
            edge = self.np_random.integers(4)
            if edge == 0: # top
                start_pos = (self.np_random.uniform(0, self.WIDTH), -20)
                target_pos = (self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8), self.HEIGHT * 0.5)
            elif edge == 1: # bottom
                start_pos = (self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
                target_pos = (self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8), self.HEIGHT * 0.5)
            elif edge == 2: # left
                start_pos = (-20, self.np_random.uniform(0, self.HEIGHT))
                target_pos = (self.WIDTH * 0.5, self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8))
            else: # right
                start_pos = (self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
                target_pos = (self.WIDTH * 0.5, self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8))

            angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
            vel = (math.cos(angle) * enemy_speed, math.sin(angle) * enemy_speed)
            self.enemies.append({'pos': start_pos, 'vel': vel, 'damage': enemy_damage})
        # sfx: new_wave_starts

    def _calculate_wall_integrity(self):
        if self.total_max_tile_health == 0: return 0
        current_health = sum(tile['health'] for tile in self.tiles.values())
        return max(0, current_health / self.total_max_tile_health)

    def _create_particles(self, pos, color, count, p_type='burst'):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            elif p_type == 'impact':
                angle = self.np_random.uniform(math.pi * 0.1, math.pi * 0.9) # Upward explosion
                speed = self.np_random.uniform(1, 3)
                vel = (math.cos(angle), -math.sin(angle) * speed)
            
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos, 'vel': vel, 'life': life, 'max_life': life,
                'radius': self.np_random.uniform(2, 5), 'color': color
            })

    def _draw_glowing_circle(self, surface, color, center, radius, alpha):
        if radius <= 0: return
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), radius)
        surface.blit(surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _axial_to_pixel(self, q, r):
        x = self.HEX_RADIUS * (3/2 * q) + self.HEX_OFFSET_X
        y = self.HEX_RADIUS * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self.HEX_OFFSET_Y
        return int(x), int(y)

    def _pixel_to_axial(self, x, y):
        x -= self.HEX_OFFSET_X
        y -= self.HEX_OFFSET_Y
        q = (2/3 * x) / self.HEX_RADIUS
        r = (-1/3 * x + math.sqrt(3)/3 * y) / self.HEX_RADIUS
        return self._axial_round(q, r)

    def _axial_round(self, q, r):
        s = -q - r
        q_r, r_r, s_r = round(q), round(r), round(s)
        q_diff, r_diff, s_diff = abs(q_r - q), abs(r_r - r), abs(s_r - s)
        if q_diff > r_diff and q_diff > s_diff:
            q_r = -r_r - s_r
        elif r_diff > s_diff:
            r_r = -q_r - s_r
        return q_r, r_r
    
    def _draw_hexagon(self, surface, color, center, radius):
        points = [
            (center[0] + radius * math.cos(math.pi/3 * i), 
             center[1] + radius * math.sin(math.pi/3 * i))
            for i in range(6)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_hexagon_outline(self, surface, color, center, radius, width=1):
        points = [
            (center[0] + radius * math.cos(math.pi/3 * i),
             center[1] + radius * math.sin(math.pi/3 * i))
            for i in range(6)
        ]
        pygame.draw.aalines(surface, color, True, points, width)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and visualization.
    # It will not be executed by the autograder, but is useful for testing.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac', etc. depending on your OS

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tessellation Defense")
    clock = pygame.time.Clock()

    movement = 0
    space_held = 0
    shift_held = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Manual controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()