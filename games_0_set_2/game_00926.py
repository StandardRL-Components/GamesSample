
# Generated: 2025-08-27T15:14:43.893578
# Source Brief: brief_00926.md
# Brief Index: 926

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to place a short-range, "
        "high-damage tower. Press shift to place a long-range, low-damage tower."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing towers along their path. "
        "Earn energy by defeating enemies to build more defenses."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_PATH_BORDER = (50, 65, 80)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_TOWER_1 = (0, 130, 200) # Blue - high damage
        self.COLOR_TOWER_2 = (245, 130, 48) # Orange - long range
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.COLOR_HEALTH_BAR = (60, 180, 75)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0)
        self.COLOR_ENERGY = (255, 215, 0)

        # Game parameters
        self.BASE_MAX_HEALTH = 100
        self.ENEMY_HEALTH = 10
        self.ENEMY_SPEED = 0.8
        self.ENEMY_DAMAGE = 10
        self.ENEMY_KILL_ENERGY = 15
        self.STARTING_ENERGY = 60
        self.ENEMY_SPAWN_RATE_INITIAL = 1.0 # enemies per second
        self.ENEMY_SPAWN_RAMP_INTERVAL = 10 * self.FPS # every 10 seconds
        self.ENEMY_SPAWN_RAMP_AMOUNT = 0.2 # increase by this many enemies/sec

        # Tower Type 1: Short Range, High Damage
        self.TOWER_1_COST = 25
        self.TOWER_1_RANGE = 70
        self.TOWER_1_DAMAGE = 4
        self.TOWER_1_FIRE_RATE = 0.5 * self.FPS # shots per second

        # Tower Type 2: Long Range, Low Damage
        self.TOWER_2_COST = 20
        self.TOWER_2_RANGE = 120
        self.TOWER_2_DAMAGE = 2
        self.TOWER_2_FIRE_RATE = 0.3 * self.FPS

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Path and Tower Placement Setup ---
        self.path_waypoints = [
            pygame.Vector2(-20, 100), pygame.Vector2(150, 100),
            pygame.Vector2(150, 300), pygame.Vector2(450, 300),
            pygame.Vector2(450, 50), pygame.Vector2(self.WIDTH + 20, 50)
        ]
        self.tower_placement_zones = [
            (80, 160), (220, 160), (220, 240), (380, 240), (380, 110), (520, 110),
            (80, 40), (220, 40), (300, 360), (400, 360)
        ]
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.energy = 0
        self.cursor_zone_index = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemy_spawn_timer = 0
        self.current_spawn_rate_hz = 0
        self.action_cooldown = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.BASE_MAX_HEALTH
        self.energy = self.STARTING_ENERGY
        
        self.cursor_zone_index = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_spawn_rate_hz = self.ENEMY_SPAWN_RATE_INITIAL
        self.enemy_spawn_timer = 1.0 / self.current_spawn_rate_hz * self.FPS
        
        self.action_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.base_health <= 0 or self.steps >= self.MAX_STEPS
        
        if not self.game_over:
            # --- Update Timers ---
            self.steps += 1
            if self.action_cooldown > 0:
                self.action_cooldown -= 1

            # --- Handle Input ---
            reward += self._handle_actions(action)
            
            # --- Update Game Logic ---
            self._update_spawner()
            reward += self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
        
        # --- Check Termination ---
        terminated = self.base_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # First frame of termination
            if self.base_health > 0: # Win condition
                self.game_won = True
                reward += 100
            else: # Lose condition
                self.game_won = False
                reward -= 100
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action
        
        # --- Cursor Movement ---
        if movement != 0:
            self._move_cursor(movement)

        # --- Tower Placement ---
        if self.action_cooldown == 0:
            placed_tower = False
            current_pos = self.tower_placement_zones[self.cursor_zone_index]
            is_occupied = any(t['pos'] == current_pos for t in self.towers)

            if not is_occupied:
                if space_held and self.energy >= self.TOWER_1_COST:
                    self.energy -= self.TOWER_1_COST
                    self.towers.append({
                        'pos': current_pos, 'type': 1, 'range': self.TOWER_1_RANGE,
                        'damage': self.TOWER_1_DAMAGE, 'cooldown': 0,
                        'fire_rate': self.TOWER_1_FIRE_RATE, 'angle': -90
                    })
                    placed_tower = True
                elif shift_held and self.energy >= self.TOWER_2_COST:
                    self.energy -= self.TOWER_2_COST
                    self.towers.append({
                        'pos': current_pos, 'type': 2, 'range': self.TOWER_2_RANGE,
                        'damage': self.TOWER_2_DAMAGE, 'cooldown': 0,
                        'fire_rate': self.TOWER_2_FIRE_RATE, 'angle': -90
                    })
                    placed_tower = True
            
            if placed_tower:
                self.action_cooldown = 10 # 1/6th of a second cooldown
                # sfx: place_tower.wav
                for _ in range(20):
                    self.particles.append(self._create_spark(current_pos, self.COLOR_TOWER_1 if space_held else self.COLOR_TOWER_2))
                return 0.5 # Small reward for placing a tower
        return 0

    def _move_cursor(self, direction):
        if self.action_cooldown > 0: return

        current_pos = pygame.Vector2(self.tower_placement_zones[self.cursor_zone_index])
        best_zone_idx = -1
        min_dist_sq = float('inf')

        for i, zone_pos_tuple in enumerate(self.tower_placement_zones):
            if i == self.cursor_zone_index:
                continue
            
            zone_pos = pygame.Vector2(zone_pos_tuple)
            vec_to_zone = zone_pos - current_pos

            if vec_to_zone.length_squared() == 0: continue

            is_in_direction = False
            if direction == 1: # Up
                is_in_direction = vec_to_zone.y < 0 and abs(vec_to_zone.y) > abs(vec_to_zone.x)
            elif direction == 2: # Down
                is_in_direction = vec_to_zone.y > 0 and abs(vec_to_zone.y) > abs(vec_to_zone.x)
            elif direction == 3: # Left
                is_in_direction = vec_to_zone.x < 0 and abs(vec_to_zone.x) > abs(vec_to_zone.y)
            elif direction == 4: # Right
                is_in_direction = vec_to_zone.x > 0 and abs(vec_to_zone.x) > abs(vec_to_zone.y)
            
            if is_in_direction:
                dist_sq = vec_to_zone.length_squared()
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_zone_idx = i
        
        if best_zone_idx != -1:
            self.cursor_zone_index = best_zone_idx
            self.action_cooldown = 5 # 1/12th of a second cooldown

    def _update_spawner(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.ENEMY_SPAWN_RAMP_INTERVAL == 0:
            self.current_spawn_rate_hz += self.ENEMY_SPAWN_RAMP_AMOUNT
        
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemies.append({
                'pos': pygame.Vector2(self.path_waypoints[0]),
                'health': self.ENEMY_HEALTH,
                'max_health': self.ENEMY_HEALTH,
                'speed': self.ENEMY_SPEED * (1 + self.np_random.uniform(-0.1, 0.1)),
                'path_index': 1,
            })
            self.enemy_spawn_timer = (1.0 / self.current_spawn_rate_hz) * self.FPS
    
    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            else:
                target = None
                min_dist_sq = tower['range'] ** 2
                
                for enemy in self.enemies:
                    dist_sq = (enemy['pos'] - pygame.Vector2(tower['pos'])).length_squared()
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target = enemy
                
                if target:
                    # sfx: shoot_laser.wav
                    tower['cooldown'] = tower['fire_rate']
                    self.projectiles.append({
                        'pos': pygame.Vector2(tower['pos']),
                        'target': target,
                        'damage': tower['damage'],
                        'speed': 10
                    })
                    # Rotate tower to face target
                    dx = target['pos'].x - tower['pos'][0]
                    dy = target['pos'].y - tower['pos'][1]
                    tower['angle'] = math.degrees(math.atan2(-dx, -dy))

        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            direction = (target_pos - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if (proj['pos'] - target_pos).length_squared() < 10**2:
                # sfx: hit_enemy.wav
                proj['target']['health'] -= proj['damage']
                self.projectiles.remove(proj)
                reward += 0.1 # Reward for hitting
                self.score += 1
                for _ in range(5):
                    self.particles.append(self._create_spark(target_pos, self.COLOR_PROJECTILE))
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                # sfx: enemy_explode.wav
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY)
                self.enemies.remove(enemy)
                self.energy += self.ENEMY_KILL_ENERGY
                reward += 1 # Reward for kill
                self.score += 10
                continue
            
            if enemy['path_index'] >= len(self.path_waypoints):
                # sfx: base_damage.wav
                self.base_health -= self.ENEMY_DAMAGE
                self._create_explosion(enemy['pos'], self.COLOR_BASE)
                self.enemies.remove(enemy)
                continue
            
            target_waypoint = self.path_waypoints[enemy['path_index']]
            direction = (target_waypoint - enemy['pos'])
            
            if direction.length_squared() < (enemy['speed'] ** 2):
                enemy['path_index'] += 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']
        return reward
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, 34)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 30)
        
        # Render Base
        base_pos = self.path_waypoints[-1]
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(base_pos.x - 20), int(base_pos.y)), 15)

        # Render Tower Placement Zones
        for i, pos in enumerate(self.tower_placement_zones):
            color = (60, 70, 80) if i != self.cursor_zone_index else (150, 150, 150)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, color)

        # Render Towers
        for tower in self.towers:
            color = self.COLOR_TOWER_1 if tower['type'] == 1 else self.COLOR_TOWER_2
            p = pygame.Vector2(tower['pos'])
            
            # Triangle points
            p1 = pygame.Vector2(0, -12).rotate(tower['angle']) + p
            p2 = pygame.Vector2(-8, 8).rotate(tower['angle']) + p
            p3 = pygame.Vector2(8, 8).rotate(tower['angle']) + p
            
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = 8
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            # Health bar for enemy
            health_pct = enemy['health'] / enemy['max_health']
            if health_pct < 1.0:
                pygame.draw.rect(self.screen, (0,0,0), (rect.left, rect.top - 5, rect.width, 3))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (rect.left, rect.top - 5, int(rect.width * health_pct), 3))


        # Render Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))
            
        # Render Cursor
        cursor_pos = self.tower_placement_zones[self.cursor_zone_index]
        is_occupied = any(t['pos'] == cursor_pos for t in self.towers)
        color = self.COLOR_CURSOR_INVALID if is_occupied else self.COLOR_CURSOR
        
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, cursor_pos[0], cursor_pos[1], 20, color)
        pygame.gfxdraw.aacircle(temp_surf, cursor_pos[0], cursor_pos[1], 20, color)
        self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.base_health / self.BASE_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.WIDTH // 2 - bar_width // 2, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.WIDTH // 2 - bar_width // 2, 10, int(bar_width * health_pct), 20))
        health_text = self.font_ui.render(f"BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH // 2 - bar_width // 2 - health_text.get_width() - 10, 11))

        # Time
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 11))

        # Energy
        energy_text = self.font_ui.render(f"ENERGY: {self.energy}", True, self.COLOR_ENERGY)
        self.screen.blit(energy_text, (10, 35))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 11))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "VICTORY!" if self.game_won else "GAME OVER"
            status_text = self.font_game_over.render(status_text_str, True, self.COLOR_TEXT)
            text_rect = status_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "base_health": self.base_health,
        }
        
    def _create_explosion(self, pos, color):
        for _ in range(50):
            self.particles.append(self._create_spark(pos, color, 3))

    def _create_spark(self, pos, color, speed_mult=1.0):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3) * speed_mult
        lifespan = self.np_random.integers(15, 40)
        return {
            'pos': pygame.Vector2(pos),
            'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'color': color,
            'size': self.np_random.integers(1, 4)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `auto_advance` to True in the class to see the game run in real time
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0

    while running:
        # --- Human Input Handling ---
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()