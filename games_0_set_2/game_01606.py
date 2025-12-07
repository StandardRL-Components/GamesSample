
# Generated: 2025-08-27T17:40:07.349957
# Source Brief: brief_01606.md
# Brief Index: 1606

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press space to build a tower on the selected tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-view tower defense game. Defend your base from waves of geometric enemies by placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10 * 60 * self.FPS # 10 minutes max
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (40, 44, 52)
        self.COLOR_SLOT = (60, 66, 78)
        self.COLOR_SLOT_SELECTED = (224, 108, 117)
        self.COLOR_BASE = (148, 198, 107)
        self.COLOR_TOWER = (97, 175, 239)
        self.COLOR_ENEMY = (224, 108, 117)
        self.COLOR_PROJECTILE = (229, 192, 123)
        self.COLOR_TEXT = (200, 205, 215)
        self.COLOR_HEALTH_GREEN = (80, 200, 120)
        self.COLOR_HEALTH_RED = (200, 80, 80)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)

        # Game state variables are initialized in reset()
        self.path_waypoints = []
        self.tower_slots = []
        self._define_layout()
        
        self.reset()
        
        # Used to handle single-press actions
        self.last_movement_action = 0
        self.last_space_held = False

        self.validate_implementation()

    def _define_layout(self):
        # Define the enemy path
        self.path_waypoints = [
            (-20, 100), (100, 100), (100, 300), (300, 300), (300, 100),
            (500, 100), (500, 300), (self.WIDTH + 20, 300)
        ]
        
        # Define tower placement slots
        slot_size = 40
        for row in range(3):
            for col in range(5):
                x = 140 + col * (slot_size + 15)
                y = 150 + row * (slot_size + 20)
                if row % 2 == 1:
                    x += 20
                self.tower_slots.append({'pos': (x, y), 'occupied': False})
        self.selector_grid_dims = (5, 3) # (cols, rows)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.victory = False

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 150

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = self.FPS * 5  # Time until first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0

        self.selector_pos = [0, 0] # col, row

        # Reset tower slot occupancy
        for slot in self.tower_slots:
            slot['occupied'] = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = -0.001 # Small penalty for existing

        self._handle_input(action)
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        self._check_collisions()
        self._cleanup()

        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle selector movement (on press, not hold)
        if movement != 0 and movement != self.last_movement_action:
            if movement == 1: # Up
                self.selector_pos[1] = (self.selector_pos[1] - 1) % self.selector_grid_dims[1]
            elif movement == 2: # Down
                self.selector_pos[1] = (self.selector_pos[1] + 1) % self.selector_grid_dims[1]
            elif movement == 3: # Left
                self.selector_pos[0] = (self.selector_pos[0] - 1) % self.selector_grid_dims[0]
            elif movement == 4: # Right
                self.selector_pos[0] = (self.selector_pos[0] + 1) % self.selector_grid_dims[0]
        
        # Handle tower placement (on press, not hold)
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_movement_action = movement
        self.last_space_held = space_held

    def _place_tower(self):
        tower_cost = 100
        if self.resources >= tower_cost:
            index = self.selector_pos[1] * self.selector_grid_dims[0] + self.selector_pos[0]
            if 0 <= index < len(self.tower_slots) and not self.tower_slots[index]['occupied']:
                # sfx: tower_build.wav
                self.resources -= tower_cost
                slot = self.tower_slots[index]
                slot['occupied'] = True
                self.towers.append({
                    'pos': slot['pos'],
                    'range': 100,
                    'cooldown': 0,
                    'fire_rate': 20, # frames between shots
                    'damage': 25
                })
                # Visual feedback for successful build
                self._create_particles(slot['pos'], self.COLOR_TOWER, 20, 5, 20)
        else:
            # sfx: error.wav
            # Visual feedback for failure
            index = self.selector_pos[1] * self.selector_grid_dims[0] + self.selector_pos[0]
            if 0 <= index < len(self.tower_slots):
                pos = self.tower_slots[index]['pos']
                self._create_particles(pos, self.COLOR_ENEMY, 10, 2, 10)


    def _update_waves(self):
        if self.current_wave >= self.MAX_WAVES and not self.enemies:
            return

        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.enemies_to_spawn == 0 and self.current_wave < self.MAX_WAVES:
            self.current_wave += 1
            self.enemies_to_spawn = 18 + 2 * self.current_wave
            self.spawn_timer = 0
            self.wave_timer = self.FPS * 15 # Cooldown between waves

        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.spawn_timer = self.FPS * 0.5 # Interval between enemies

    def _spawn_enemy(self):
        # sfx: enemy_spawn.wav
        health = 50 + self.current_wave * 10
        self.enemies.append({
            'pos': list(self.path_waypoints[0]),
            'health': health,
            'max_health': health,
            'speed': 1.0 + self.current_wave * 0.1,
            'waypoint_idx': 1,
            'damage': 10,
            'bounty': 10 + self.current_wave
        })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    # sfx: tower_shoot.wav
                    self.projectiles.append({
                        'pos': list(tower['pos']),
                        'target': target,
                        'speed': 8,
                        'damage': tower['damage']
                    })
                    tower['cooldown'] = tower['fire_rate']
                    # Muzzle flash
                    self._create_particles(tower['pos'], self.COLOR_PROJECTILE, 5, 3, 5)

    def _find_target(self, tower):
        best_target = None
        max_dist_traveled = -1
        
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
            if dist <= tower['range']:
                # Target enemy that is furthest along the path
                dist_traveled = self._get_enemy_distance_traveled(enemy)
                if dist_traveled > max_dist_traveled:
                    max_dist_traveled = dist_traveled
                    best_target = enemy
        return best_target

    def _get_enemy_distance_traveled(self, enemy):
        dist = 0
        for i in range(enemy['waypoint_idx']):
            dist += math.hypot(self.path_waypoints[i+1][0] - self.path_waypoints[i][0], self.path_waypoints[i+1][1] - self.path_waypoints[i][1])
        
        p1 = self.path_waypoints[enemy['waypoint_idx']-1]
        dist += math.hypot(enemy['pos'][0] - p1[0], enemy['pos'][1] - p1[1])
        return dist

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                continue
            
            target_pos = self.path_waypoints[enemy['waypoint_idx']]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['waypoint_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']

    def _update_projectiles(self):
        for p in self.projectiles:
            if p['target'] in self.enemies:
                target_pos = p['target']['pos']
                dx = target_pos[0] - p['pos'][0]
                dy = target_pos[1] - p['pos'][1]
                dist = math.hypot(dx, dy)
                if dist < p['speed']:
                    p['pos'] = list(target_pos)
                else:
                    p['pos'][0] += (dx / dist) * p['speed']
                    p['pos'][1] += (dy / dist) * p['speed']
            else: # Target is gone, fly straight
                p['target'] = None
                p['pos'][1] -= p['speed'] # Fly off screen

    def _update_particles(self):
        for particle in self.particles:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['vel'][1] += 0.1 # Gravity
            particle['life'] -= 1
            particle['radius'] = max(0, particle['radius'] - 0.1)

    def _check_collisions(self):
        # Projectile-Enemy
        for p in list(self.projectiles):
            for e in list(self.enemies):
                if math.hypot(p['pos'][0] - e['pos'][0], p['pos'][1] - e['pos'][1]) < 10:
                    # sfx: enemy_hit.wav
                    e['health'] -= p['damage']
                    self.reward_this_step += 0.1
                    self.score += 1
                    self._create_particles(e['pos'], self.COLOR_PROJECTILE, 10, 3, 10)
                    if p in self.projectiles: self.projectiles.remove(p)
                    break
        
        # Enemy-Base
        for e in list(self.enemies):
            if e['waypoint_idx'] >= len(self.path_waypoints):
                # sfx: base_damage.wav
                self.base_health = max(0, self.base_health - e['damage'])
                if e in self.enemies: self.enemies.remove(e)
                # Create a damage particle effect on the base
                self._create_particles((self.WIDTH-25, self.HEIGHT-50), self.COLOR_ENEMY, 30, 8, 25)


    def _cleanup(self):
        # Dead enemies
        for e in list(self.enemies):
            if e['health'] <= 0:
                # sfx: enemy_die.wav
                self.reward_this_step += 1.0
                self.score += 10
                self.resources += e['bounty']
                self._create_particles(e['pos'], self.COLOR_ENEMY, 30, 4, 20)
                self.enemies.remove(e)
        
        # Used projectiles and old particles
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.reward_this_step -= 100
            self.score -= 1000
            self.game_over = True
            return True
        
        if self.current_wave >= self.MAX_WAVES and not self.enemies and self.enemies_to_spawn == 0:
            self.reward_this_step += 100
            self.score += 1000
            self.game_over = True
            self.victory = True
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
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)
        pygame.draw.lines(self.screen, tuple(c*1.1 for c in self.COLOR_PATH), False, self.path_waypoints, 2)

        # Draw tower slots and selector
        for i, slot in enumerate(self.tower_slots):
            is_selected = (i == self.selector_pos[1] * self.selector_grid_dims[0] + self.selector_pos[0])
            color = self.COLOR_SLOT_SELECTED if is_selected else self.COLOR_SLOT
            pygame.draw.rect(self.screen, color, (*slot['pos'], 20, 20), 2, border_radius=3)

        # Draw base
        base_rect = pygame.Rect(self.WIDTH - 25, self.HEIGHT - 100, 25, 100)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            x, y = int(tower['pos'][0])+10, int(tower['pos'][1])+10
            points = [(x, y-10), (x-8, y+6), (x+8, y+6)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TOWER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TOWER)

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (x - 10, y - 15, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (x - 10, y - 15, int(20 * health_pct), 3))

        # Draw projectiles
        for p in self.projectiles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, self.COLOR_PROJECTILE)

        # Draw particles
        for particle in self.particles:
            x, y = int(particle['pos'][0]), int(particle['pos'][1])
            radius = int(particle['radius'])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, particle['color'])
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, particle['color'])

    def _render_ui(self):
        # Wave number
        wave_text = self.font_medium.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Resources
        resource_text = self.font_medium.render(f"Resources: {self.resources}", True, self.COLOR_PROJECTILE)
        self.screen.blit(resource_text, (10, self.HEIGHT - 30))

        # Base Health Bar
        health_pct = self.base_health / self.max_base_health
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (self.WIDTH - bar_width - 10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (self.WIDTH - bar_width - 10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render(f"{self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - bar_width / 2 - 20, 12))

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.victory else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_BASE if self.victory else self.COLOR_ENEMY)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
        }

    def _create_particles(self, pos, color, count, speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag],
                'life': random.randint(life // 2, life),
                'color': color,
                'radius': random.uniform(2, 5)
            })

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Input Handling ---
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
            
        action = [movement, space_pressed, shift_pressed]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()