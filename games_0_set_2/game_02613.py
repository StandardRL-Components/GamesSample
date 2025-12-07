
# Generated: 2025-08-28T05:26:16.539660
# Source Brief: brief_02613.md
# Brief Index: 2613

        
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
        "Controls: ←↑→↓ to select a tower position. Press space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_PATH = (70, 70, 80)
    COLOR_BASE = (60, 180, 60)
    COLOR_TOWER_ZONE = (50, 50, 60)
    COLOR_TOWER_ZONE_ACTIVE = (200, 200, 100)
    COLOR_ENEMY = (210, 50, 50)
    COLOR_TOWER = (50, 150, 250)
    COLOR_PROJECTILE = (255, 220, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_GOLD = (255, 215, 0)
    
    # Game Parameters
    INITIAL_GOLD = 30
    TOWER_COST = 10
    TOWER_RANGE = 85
    TOWER_COOLDOWN = 30  # frames
    PROJECTILE_SPEED = 8
    
    ENEMY_START_HEALTH = 5
    ENEMY_SPEED = 1.2
    ENEMY_KILL_GOLD = 5
    TOTAL_ENEMIES_TO_WIN = 20

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self._define_world()
        
        # Etc...
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Initialize state variables
        self.reset()
    
    def _define_world(self):
        self.path = [
            pygame.Vector2(-20, 200),
            pygame.Vector2(100, 200),
            pygame.Vector2(100, 100),
            pygame.Vector2(540, 100),
            pygame.Vector2(540, 300),
            pygame.Vector2(self.SCREEN_WIDTH + 20, 300)
        ]
        self.base_pos = pygame.Vector2(540, 300)
        self.base_radius = 25

        self.placement_zones = [
            pygame.Vector2(100, 150),
            pygame.Vector2(220, 150),
            pygame.Vector2(320, 40),
            pygame.Vector2(440, 150),
            pygame.Vector2(540, 200),
            pygame.Vector2(440, 250),
            pygame.Vector2(320, 300),
            pygame.Vector2(220, 250),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.gold = self.INITIAL_GOLD
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_index = 0
        self.prev_space_held = False
        self.last_movement_action = 0
        
        self.total_enemies_defeated = 0
        
        self.wave_num = 0
        self.wave_enemies_to_spawn = 0
        self.wave_spawn_cooldown = 0
        self.inter_wave_timer = 90 # Time before first wave

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        
        self._handle_actions(movement, space_held)
        
        if not self.game_over:
            self._update_waves()
            self._update_towers()
            projectile_reward = self._update_projectiles()
            enemy_reward = self._update_enemies()
            reward += projectile_reward + enemy_reward

        self._update_particles()
        
        # Update game logic
        self.steps += 1
        
        win = self.total_enemies_defeated >= self.TOTAL_ENEMIES_TO_WIN and not self.enemies
        loss = self.game_over
        timeout = self.steps >= self.MAX_STEPS
        
        terminated = win or loss or timeout
        
        if win and not loss:
            reward += 50
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_actions(self, movement, space_held):
        # Discrete movement for cursor
        if movement != 0 and movement != self.last_movement_action:
            if movement in [1, 3]: # Up or Left
                self.cursor_index = (self.cursor_index - 1) % len(self.placement_zones)
            elif movement in [2, 4]: # Down or Right
                self.cursor_index = (self.cursor_index + 1) % len(self.placement_zones)
        self.last_movement_action = movement

        # Discrete press for placing tower
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held

    def _place_tower(self):
        if self.gold >= self.TOWER_COST:
            pos = self.placement_zones[self.cursor_index]
            is_occupied = any(tower['pos'] == pos for tower in self.towers)
            
            if not is_occupied:
                self.gold -= self.TOWER_COST
                self.towers.append({
                    'pos': pos,
                    'cooldown': 0,
                    'angle': 0
                })
                # sfx: tower_place.wav

    def _update_waves(self):
        if self.total_enemies_defeated + len(self.enemies) >= self.TOTAL_ENEMIES_TO_WIN:
            return

        if self.wave_enemies_to_spawn == 0:
            if self.inter_wave_timer > 0:
                self.inter_wave_timer -= 1
            else:
                self._start_next_wave()
        else:
            if self.wave_spawn_cooldown > 0:
                self.wave_spawn_cooldown -= 1
            else:
                self._spawn_enemy()
                self.wave_enemies_to_spawn -= 1
                self.wave_spawn_cooldown = 20 # Time between enemies in a wave

    def _start_next_wave(self):
        self.wave_num += 1
        num_enemies = min(self.wave_num, 4)
        remaining_to_win = self.TOTAL_ENEMIES_TO_WIN - (self.total_enemies_defeated + len(self.enemies))
        self.wave_enemies_to_spawn = min(num_enemies, remaining_to_win)
        self.inter_wave_timer = 150 # Time between waves

    def _spawn_enemy(self):
        health = self.ENEMY_START_HEALTH
        if self.wave_num > 5:
            health += (self.wave_num - 5) # Per brief: +1 health every wave *after* wave 5
        
        self.enemies.append({
            'pos': self.path[0].copy(),
            'health': health,
            'max_health': health,
            'path_index': 0,
            'distance_on_segment': 0
        })

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            target = None
            min_dist = float('inf')
            
            for enemy in self.enemies:
                dist = tower['pos'].distance_to(enemy['pos'])
                if dist < self.TOWER_RANGE and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                # sfx: tower_shoot.wav
                self.projectiles.append({
                    'pos': tower['pos'].copy(),
                    'target': target,
                    'speed': self.PROJECTILE_SPEED
                })
                tower['cooldown'] = self.TOWER_COOLDOWN
                
                dx = target['pos'].x - tower['pos'].x
                dy = target['pos'].y - tower['pos'].y
                tower['angle'] = math.degrees(math.atan2(-dy, dx))

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            if p['target'] not in self.enemies:
                projectiles_to_remove.append(i)
                continue

            direction = (p['target']['pos'] - p['pos']).normalize()
            p['pos'] += direction * p['speed']

            if p['pos'].distance_to(p['target']['pos']) < 8:
                # sfx: projectile_hit.wav
                p['target']['health'] -= 1
                reward += 0.1
                projectiles_to_remove.append(i)
                self._create_particles(p['pos'], self.COLOR_PROJECTILE, 5)
        
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy['health'] <= 0:
                # sfx: enemy_die.wav
                reward += 1
                self.gold += self.ENEMY_KILL_GOLD
                self.total_enemies_defeated += 1
                enemies_to_remove.append(i)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 2.5)
                continue

            if enemy['path_index'] >= len(self.path) - 1:
                if not self.game_over: # Only trigger once
                    # sfx: game_over.wav
                    reward -= 50
                    self.game_over = True
                enemies_to_remove.append(i)
                continue
            
            start_node = self.path[enemy['path_index']]
            end_node = self.path[enemy['path_index'] + 1]
            
            segment_vec = end_node - start_node
            segment_len = segment_vec.length()

            if segment_len > 0:
                enemy['distance_on_segment'] += self.ENEMY_SPEED
                
                if enemy['distance_on_segment'] >= segment_len:
                    enemy['path_index'] += 1
                    enemy['distance_on_segment'] = 0
                else:
                    enemy['pos'] = start_node + segment_vec * (enemy['distance_on_segment'] / segment_len)
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
            
        return reward

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _create_particles(self, pos, color, count, speed_scale=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_scale
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(10, 20),
                'color': color,
                'radius': random.uniform(1, 3)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(int(p.x), int(p.y)) for p in self.path], 10)
        
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos.x), int(self.base_pos.y), self.base_radius, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos.x), int(self.base_pos.y), self.base_radius, self.COLOR_BASE)
        
        cursor_pos = self.placement_zones[self.cursor_index]
        for i, pos in enumerate(self.placement_zones):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            color = self.COLOR_TOWER_ZONE_ACTIVE if i == self.cursor_index else self.COLOR_TOWER_ZONE
            if is_occupied:
                color = (100,20,20)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, color)
        
        pygame.draw.circle(self.screen, self.COLOR_TOWER_ZONE_ACTIVE, (int(cursor_pos.x), int(cursor_pos.y)), 12, 2)

        for tower in self.towers:
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            range_color = (*self.COLOR_TOWER, 50)
            s = pygame.Surface((self.TOWER_RANGE*2, self.TOWER_RANGE*2), pygame.SRCALPHA)
            pygame.draw.circle(s, range_color, (self.TOWER_RANGE, self.TOWER_RANGE), self.TOWER_RANGE)
            self.screen.blit(s, (pos[0] - self.TOWER_RANGE, pos[1] - self.TOWER_RANGE))

            rect = pygame.Rect(pos[0] - 8, pos[1] - 8, 16, 16)
            pygame.draw.rect(self.screen, self.COLOR_TOWER, rect, border_radius=3)
            
            barrel_len = 12
            end_x = pos[0] + barrel_len * math.cos(math.radians(tower['angle']))
            end_y = pos[1] - barrel_len * math.sin(math.radians(tower['angle']))
            pygame.draw.line(self.screen, (200,200,220), pos, (end_x, end_y), 4)

        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            radius = 7
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, tuple(int(c*0.8) for c in self.COLOR_ENEMY))
            
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            bar_height = 4
            health_bar_rect = pygame.Rect(pos[0] - bar_width/2, pos[1] - radius - bar_height - 2, bar_width, bar_height)
            pygame.draw.rect(self.screen, (80,0,0), health_bar_rect)
            pygame.draw.rect(self.screen, (0,200,0), (health_bar_rect.x, health_bar_rect.y, bar_width * health_ratio, bar_height))

        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(p['radius']),int(p['radius'])), int(p['radius']))
            self.screen.blit(s, (pos[0]-p['radius'], pos[1]-p['radius']))

    def _render_ui(self):
        gold_text = self.font_small.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))
        
        enemies_left = self.TOTAL_ENEMIES_TO_WIN - self.total_enemies_defeated
        wave_text = self.font_small.render(f"WAVE: {self.wave_num} | ENEMIES LEFT: {enemies_left}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 30))

        msg = None
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_ENEMY
        elif self.total_enemies_defeated >= self.TOTAL_ENEMIES_TO_WIN and not self.enemies:
            msg = "YOU WIN!"
            color = self.COLOR_BASE

        if msg:
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_num,
            "enemies_defeated": self.total_enemies_defeated,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()