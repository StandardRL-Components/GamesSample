
# Generated: 2025-08-28T05:46:47.699155
# Source Brief: brief_05690.md
# Brief Index: 5690

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to place a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically "
        "placing defensive towers in an isometric grid."
    )

    auto_advance = True

    # --- Constants ---
    # Game settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 40, 20
    MAX_STEPS = 30000  # Approx 16 minutes at 30fps
    FPS = 30

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_PATH = (70, 60, 50)
    COLOR_BASE = (100, 80, 60)
    COLOR_BASE_HEALTH = (70, 180, 70)
    COLOR_BASE_HEALTH_BG = (180, 70, 70)
    COLOR_CURSOR_VALID = (100, 255, 100, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100, 100)

    COLOR_ENEMY = (210, 50, 50)
    COLOR_ENEMY_HEALTH = (200, 60, 60)

    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_ACCENT = (255, 200, 0)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    # --- Tower Definitions ---
    TOWER_TYPES = [
        {
            "name": "Cannon",
            "cost": 100, "damage": 25, "range": 3.5, "fire_rate": 1.0, # seconds
            "color": (0, 150, 255), "projectile_speed": 8, "projectile_size": 3,
        },
        {
            "name": "Machine Gun",
            "cost": 150, "damage": 8, "range": 2.5, "fire_rate": 0.2,
            "color": (255, 150, 0), "projectile_speed": 12, "projectile_size": 2,
        },
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.wave_state = "INTER_WAVE" # or "WAVE_IN_PROGRESS"
        self.wave_timer = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [0,0]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # Define path and buildable areas
        self._define_layout()

        self.reset()
        self.validate_implementation()

    def _define_layout(self):
        self.iso_origin_x = self.SCREEN_WIDTH // 2
        self.iso_origin_y = 100
        
        self.path_coords = [
            (0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (3, 2),
            (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (9, 3),
            (9, 4), (9, 5), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6),
            (14, 6), (15, 6), (16, 6), (16, 5), (16, 4), (17, 4), (18, 4), (19, 4)
        ]
        self.base_pos = (19, 3)
        self.path_coords.append(self.base_pos)

        self.grid = [["buildable" for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        for x, y in self.path_coords:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x][y] = "path"
        self.grid[self.base_pos[0]][self.base_pos[1]] = "base"

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.iso_origin_x + (grid_x - grid_y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.iso_origin_y + (grid_x + grid_y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.base_health = 1000
        self.resources = 250
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > 10:
            return
        self.wave_state = "WAVE_IN_PROGRESS"
        self.enemies_to_spawn = 5 + self.wave_number * 2
        self.spawn_timer = 0
        
    def _get_wave_enemy_stats(self):
        base_health = 50 * (1.05 ** (self.wave_number - 1))
        base_speed = 0.5 * (1.02 ** (self.wave_number - 1))
        return base_health, base_speed

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.001 # Small time penalty to encourage efficiency
        
        if not self.game_over:
            self._handle_input(action)
            self._update_wave()
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100
            else: # Loss
                reward -= 100
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower (Shift) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle.wav

        # --- Place Tower (Space) ---
        if space_held and not self.prev_space_held:
            tower_def = self.TOWER_TYPES[self.selected_tower_type]
            cx, cy = self.cursor_pos
            is_buildable = self.grid[cx][cy] == "buildable"
            
            if self.resources >= tower_def["cost"] and is_buildable:
                self.resources -= tower_def["cost"]
                self.towers.append({
                    "pos": tuple(self.cursor_pos),
                    "type": self.selected_tower_type,
                    "cooldown": 0
                })
                self.grid[cx][cy] = "occupied"
                # sfx: place_tower.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_wave(self):
        if self.wave_state == "WAVE_IN_PROGRESS":
            if self.enemies_to_spawn > 0:
                self.spawn_timer -= 1 / self.FPS
                if self.spawn_timer <= 0:
                    health, speed = self._get_wave_enemy_stats()
                    self.enemies.append({
                        "path_index": 0, "progress": 0.0, "health": health,
                        "max_health": health, "speed": speed, "pos": self.path_coords[0]
                    })
                    self.enemies_to_spawn -= 1
                    self.spawn_timer = 1.0 # Time between spawns
                    # sfx: enemy_spawn.wav
            elif not self.enemies:
                if self.wave_number >= 10:
                    self.victory = True
                else:
                    self.wave_state = "INTER_WAVE"
                    self.wave_timer = 10 # 10 seconds between waves
        
        elif self.wave_state == "INTER_WAVE":
            self.wave_timer -= 1 / self.FPS
            if self.wave_timer <= 0:
                self._start_next_wave()

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1 / self.FPS)
            if tower['cooldown'] > 0:
                continue

            tower_def = self.TOWER_TYPES[tower['type']]
            target = None
            
            # Find first enemy in range
            for enemy in self.enemies:
                ex, ey = self._iso_to_screen(enemy["pos"][0], enemy["pos"][1])
                tx, ty = self._iso_to_screen(tower["pos"][0], tower["pos"][1])
                dist = math.hypot(ex - tx, ey - ty)
                if dist <= tower_def["range"] * self.TILE_WIDTH_ISO / 2:
                    target = enemy
                    break
            
            if target:
                tower['cooldown'] = tower_def['fire_rate']
                start_pos = self._iso_to_screen(tower["pos"][0], tower["pos"][1])
                target_pos = self._iso_to_screen(target["pos"][0], target["pos"][1])
                self.projectiles.append({
                    "pos": list(start_pos), "target_pos": target_pos, "type": tower['type'], "target": target
                })
                # sfx: tower_fire.wav (differentiated by type)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p_def = self.TOWER_TYPES[p['type']]
            
            # Move towards target
            direction = np.array(p['target_pos']) - np.array(p['pos'])
            distance = np.linalg.norm(direction)
            
            if distance < p_def['projectile_speed']:
                # Hit
                reward += self._handle_hit(p)
                projectiles_to_remove.append(i)
                continue
            
            direction = direction / distance
            p['pos'] += direction * p_def['projectile_speed']
            
            # Check if target is dead or projectile is off-screen
            if p['target'] not in self.enemies or not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                projectiles_to_remove.append(i)

        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
        return reward
        
    def _handle_hit(self, projectile):
        reward = 0.1
        p_def = self.TOWER_TYPES[projectile['type']]
        projectile['target']['health'] -= p_def['damage']
        
        hit_pos = self._iso_to_screen(projectile['target']['pos'][0], projectile['target']['pos'][1])
        # sfx: projectile_hit.wav
        self._create_particles(hit_pos, p_def['color'], 5)
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy['health'] <= 0:
                enemies_to_remove.append(i)
                self.resources += 20 + self.wave_number
                reward += 1.0
                # sfx: enemy_die.wav
                death_pos = self._iso_to_screen(enemy['pos'][0], enemy['pos'][1])
                self._create_particles(death_pos, self.COLOR_ENEMY, 15, 2)
                continue

            # Move along path
            path_index = enemy['path_index']
            if path_index >= len(self.path_coords) - 1:
                # Reached base
                self.base_health -= enemy['health']
                reward -= 10
                enemies_to_remove.append(i)
                # sfx: base_damage.wav
                continue

            start_node = self.path_coords[path_index]
            end_node = self.path_coords[path_index + 1]
            segment_dist = math.hypot(end_node[0] - start_node[0], end_node[1] - start_node[1])
            
            # distance to move this frame
            move_dist = (enemy['speed'] / self.FPS) / (segment_dist if segment_dist > 0 else 1)
            enemy['progress'] += move_dist

            if enemy['progress'] >= 1.0:
                enemy['path_index'] += 1
                enemy['progress'] = 0.0
            
            # Interpolate position for smooth rendering
            p_idx = enemy['path_index']
            start_pos = self.path_coords[p_idx]
            end_pos = self.path_coords[p_idx + 1] if p_idx + 1 < len(self.path_coords) else start_pos
            
            interp_x = start_pos[0] + (end_pos[0] - start_pos[0]) * enemy['progress']
            interp_y = start_pos[1] + (end_pos[1] - start_pos[1]) * enemy['progress']
            enemy['pos'] = (interp_x, interp_y)
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        return reward
        
    def _create_particles(self, pos, color, count, speed_mult=1):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.uniform(0.2, 0.5), # seconds
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1 / self.FPS
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.base_health = 0
            return True
        if self.victory:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2),
                    (sx, sy + self.TILE_HEIGHT_ISO),
                    (sx - self.TILE_WIDTH_ISO / 2, sy + self.TILE_HEIGHT_ISO / 2)
                ]
                color = self.COLOR_PATH if self.grid[x][y] == "path" else self.COLOR_GRID
                pygame.draw.polygon(self.screen, color, points)

        # Render base
        bx, by = self._iso_to_screen(self.base_pos[0], self.base_pos[1])
        base_points = [
            (bx, by - self.TILE_HEIGHT_ISO/2),
            (bx + self.TILE_WIDTH_ISO, by),
            (bx, by + self.TILE_HEIGHT_ISO/2),
            (bx - self.TILE_WIDTH_ISO, by),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_BASE, base_points)

        # Render towers
        for tower in self.towers:
            t_def = self.TOWER_TYPES[tower['type']]
            tx, ty = self._iso_to_screen(tower['pos'][0], tower['pos'][1])
            pygame.draw.circle(self.screen, t_def['color'], (tx, ty), 8)
            pygame.draw.circle(self.screen, (255,255,255), (tx, ty), 8, 1)

        # Render enemies
        for enemy in self.enemies:
            ex, ey = self._iso_to_screen(enemy['pos'][0], enemy['pos'][1])
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (ex, ey), 6)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 12
            pygame.draw.rect(self.screen, (0,0,0), (ex - bar_w/2 -1, ey - 14, bar_w+2, 5))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (ex - bar_w/2, ey - 13, bar_w * health_pct, 3))

        # Render projectiles
        for p in self.projectiles:
            p_def = self.TOWER_TYPES[p['type']]
            pygame.draw.circle(self.screen, p_def['color'], (int(p['pos'][0]), int(p['pos'][1])), p_def['projectile_size'])

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 0.5))))
            color = (*p['color'], alpha)
            surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (2,2), 2)
            self.screen.blit(surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

        # Render cursor
        cx, cy = self.cursor_pos
        cursor_screen_x, cursor_screen_y = self._iso_to_screen(cx, cy)
        points = [
            (cursor_screen_x, cursor_screen_y),
            (cursor_screen_x + self.TILE_WIDTH_ISO / 2, cursor_screen_y + self.TILE_HEIGHT_ISO / 2),
            (cursor_screen_x, cursor_screen_y + self.TILE_HEIGHT_ISO),
            (cursor_screen_x - self.TILE_WIDTH_ISO / 2, cursor_screen_y + self.TILE_HEIGHT_ISO / 2)
        ]
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        is_valid = self.grid[cx][cy] == "buildable" and self.resources >= tower_def["cost"]
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        cursor_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.aapolygon(cursor_surf, points, color)
        pygame.gfxdraw.filled_polygon(cursor_surf, points, color)
        self.screen.blit(cursor_surf, (0,0))
    
    def _render_text(self, text, font, pos, color, shadow_color=None):
        if shadow_color:
            text_surf_s = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_s, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Base Health Bar
        health_pct = self.base_health / 1000
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH, (10, 10, bar_width * health_pct, 20))
        self._render_text(f"BASE HP", self.font_s, (15, 12), self.COLOR_TEXT)

        # Resources
        self._render_text(f"RESOURCES: {self.resources}", self.font_m, (10, 40), self.COLOR_TEXT_ACCENT, self.COLOR_TEXT_SHADOW)
        
        # Wave Info
        if self.wave_state == "INTER_WAVE":
            wave_text = f"WAVE {self.wave_number + 1} STARTING IN: {int(self.wave_timer)}"
        else:
            wave_text = f"WAVE: {self.wave_number} / 10"
        self._render_text(wave_text, self.font_m, (self.SCREEN_WIDTH - 250, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Selected Tower Info
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        self._render_text(f"Selected: {tower_def['name']}", self.font_s, (10, self.SCREEN_HEIGHT - 45), self.COLOR_TEXT)
        self._render_text(f"Cost: {tower_def['cost']}", self.font_s, (10, self.SCREEN_HEIGHT - 25), self.COLOR_TEXT)
        
        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            text_surf = self.font_l.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn,
        }

    def close(self):
        pygame.quit()
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game specific assertions
        self.reset()
        assert self.base_health == 1000
        assert self.wave_number == 1

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    
    # --- Keyboard mapping for human play ---
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4
    }
    
    while not terminated:
        # Default action is NO-OP
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        for key_code, move_action in key_map.items():
            if keys[key_code]:
                action[0] = move_action
                break
                
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over. Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()