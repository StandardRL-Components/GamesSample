import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:07:29.382397
# Source Brief: brief_01532.md
# Brief Index: 1532
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control powerful energy towers to defend against waves of geometric invaders. Manage tower aim, height, and terraform the battlefield to gain an advantage."
    user_guide = "Controls: Use arrow keys to aim the selected tower (←→) or adjust its height (↑↓). Hold Shift and use ↑↓ to terraform the ground. Press Space to cycle between towers."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.NUM_WAVES = 20
        self.TERRAIN_GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.TERRAIN_GRID_SIZE
        self.GRID_H = self.HEIGHT // self.TERRAIN_GRID_SIZE
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_TOWER = (0, 150, 255)
        self.COLOR_TOWER_SELECTED = (255, 255, 0)
        self.COLOR_PROJECTILE = (100, 255, 255)
        self.COLOR_ENEMY_A = (255, 50, 50)
        self.COLOR_ENEMY_B = (255, 150, 50)
        self.COLOR_TERRAIN_LOW = pygame.Color(30, 80, 50)
        self.COLOR_TERRAIN_HIGH = pygame.Color(100, 180, 120)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)
        
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.terrain_map = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        self.selected_tower_index = 0
        self.previous_space_state = False
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies_to_spawn = []
        self.enemy_spawn_cooldown = 0
        self.last_action = [0, 0, 0]
        
        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.towers = [
            self._create_tower(self.WIDTH * 0.2, self.HEIGHT * 0.5),
            self._create_tower(self.WIDTH * 0.5, self.HEIGHT * 0.2),
            self._create_tower(self.WIDTH * 0.8, self.HEIGHT * 0.5),
        ]
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.terrain_map = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        
        self.selected_tower_index = 0
        self.previous_space_state = False
        
        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        self.enemies_to_spawn = []
        self.enemy_spawn_cooldown = 0
        self.last_action = [0, 0, 0]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.last_action = action
        
        self._handle_input(action)
        
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        self._handle_collisions()
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        # Small penalty for existing to encourage efficiency
        self.reward_this_step -= 0.001
        
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
        shift_held = action[2] == 1
        
        # Tower selection on space PRESS
        if space_held and not self.previous_space_state:
            num_towers = len([t for t in self.towers if not t['destroyed']])
            if num_towers > 0:
                # Find next non-destroyed tower
                current_idx = self.selected_tower_index
                while True:
                    current_idx = (current_idx + 1) % len(self.towers)
                    if not self.towers[current_idx]['destroyed']:
                        self.selected_tower_index = current_idx
                        break
                    if current_idx == self.selected_tower_index: # Full loop, no active towers
                        break


        self.previous_space_state = space_held
        
        selected_tower = self.towers[self.selected_tower_index]
        if selected_tower['destroyed']:
            return

        if shift_held: # Terraforming mode
            gx = int(selected_tower['pos'][0] / self.TERRAIN_GRID_SIZE)
            gy = int(selected_tower['pos'][1] / self.TERRAIN_GRID_SIZE)
            gx = max(0, min(self.GRID_W - 1, gx))
            gy = max(0, min(self.GRID_H - 1, gy))

            if movement == 1: # Up: Raise terrain
                self.terrain_map[gy, gx] = min(5, self.terrain_map[gy, gx] + 1)
            elif movement == 2: # Down: Lower terrain
                self.terrain_map[gy, gx] = max(0, self.terrain_map[gy, gx] - 1)
        else: # Tower control mode
            if movement == 1: # Up: Increase height
                selected_tower['height'] = min(50, selected_tower['height'] + 1)
            elif movement == 2: # Down: Decrease height
                selected_tower['height'] = max(10, selected_tower['height'] - 1)
            elif movement == 3: # Left: Rotate CCW
                selected_tower['angle'] -= 0.1
            elif movement == 4: # Right: Rotate CW
                selected_tower['angle'] += 0.1
        
        selected_tower['angle'] %= (2 * math.pi)

    def _update_waves(self):
        if self.wave_timer > 0:
            self.wave_timer -= 1
            return
        
        if not self.enemies and not self.enemies_to_spawn:
            self.current_wave += 1
            if self.current_wave > self.NUM_WAVES:
                return # All waves completed
            self.reward_this_step += 100
            self.score += 1000
            
            num_enemies = 5 + self.current_wave * 2
            for _ in range(num_enemies):
                enemy_type = 'B' if self.current_wave >= 5 and self.np_random.random() > 0.7 else 'A'
                self.enemies_to_spawn.append(enemy_type)
            self.wave_timer = 300 # Time between waves
        
        if self.enemies_to_spawn:
            if self.enemy_spawn_cooldown <= 0:
                self._spawn_enemy(self.enemies_to_spawn.pop(0))
                self.enemy_spawn_cooldown = 30
            else:
                self.enemy_spawn_cooldown -= 1

    def _spawn_enemy(self, type):
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = [self.np_random.uniform(0, self.WIDTH), -20]
        elif edge == 'bottom':
            pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20]
        elif edge == 'left':
            pos = [-20, self.np_random.uniform(0, self.HEIGHT)]
        else: # right
            pos = [self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)]

        speed_multiplier = 1.0 + 0.05 * (self.current_wave // 2)
        if type == 'A':
            self.enemies.append({
                'pos': np.array(pos, dtype=float), 'type': 'A', 'size': 10,
                'health': 100, 'max_health': 100, 'speed': 1.0 * speed_multiplier,
                'target_tower': None, 'lifespan': 1000
            })
        elif type == 'B':
            self.enemies.append({
                'pos': np.array(pos, dtype=float), 'type': 'B', 'size': 15,
                'health': 200, 'max_health': 200, 'speed': 0.7 * speed_multiplier,
                'target_tower': None, 'lifespan': 1200
            })

    def _update_towers(self):
        for tower in self.towers:
            if tower['destroyed']:
                continue
            
            tower['fire_cooldown'] = max(0, tower['fire_cooldown'] - 1)
            
            # Auto-fire logic
            if tower['fire_cooldown'] == 0:
                target_enemy = None
                min_dist = float('inf')
                tower_range = 100 + tower['height'] * 2
                
                for enemy in self.enemies:
                    dist = np.linalg.norm(enemy['pos'] - tower['pos'])
                    if dist < tower_range:
                        # Check if in firing arc
                        vec_to_enemy = enemy['pos'] - tower['pos']
                        angle_to_enemy = math.atan2(vec_to_enemy[1], vec_to_enemy[0])
                        angle_diff = (tower['angle'] - angle_to_enemy + math.pi) % (2 * math.pi) - math.pi
                        if abs(angle_diff) < 0.3: # Firing arc of +/- 0.3 radians
                            if dist < min_dist:
                                min_dist = dist
                                target_enemy = enemy

                if target_enemy:
                    self.projectiles.append(self._create_projectile(tower))
                    tower['fire_cooldown'] = 40 # Reset cooldown
                    self._create_particles(tower['pos'] + np.array([math.cos(tower['angle']), math.sin(tower['angle'])]) * (tower['height'] + 5), 5, self.COLOR_PROJECTILE, 0.5)


    def _update_enemies(self):
        active_towers = [t for t in self.towers if not t['destroyed']]
        if not active_towers:
            self.enemies = []
            return

        for enemy in self.enemies:
            enemy['lifespan'] -= 1
            # Find closest tower
            if enemy['target_tower'] is None or self.towers[enemy['target_tower']]['destroyed']:
                min_dist = float('inf')
                closest_idx = -1
                for i, tower in enumerate(self.towers):
                    if not tower['destroyed']:
                        dist = np.linalg.norm(enemy['pos'] - tower['pos'])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                enemy['target_tower'] = closest_idx

            if enemy['target_tower'] is None:
                continue

            target_pos = self.towers[enemy['target_tower']]['pos']
            
            direction = target_pos - enemy['pos']
            norm = np.linalg.norm(direction)
            if norm > 1:
                direction /= norm

            current_gx = int(enemy['pos'][0] / self.TERRAIN_GRID_SIZE)
            current_gy = int(enemy['pos'][1] / self.TERRAIN_GRID_SIZE)
            
            next_pos = enemy['pos'] + direction * enemy['speed']
            next_gx = int(next_pos[0] / self.TERRAIN_GRID_SIZE)
            next_gy = int(next_pos[1] / self.TERRAIN_GRID_SIZE)

            if 0 <= current_gx < self.GRID_W and 0 <= current_gy < self.GRID_H and \
               0 <= next_gx < self.GRID_W and 0 <= next_gy < self.GRID_H:
                
                current_height = self.terrain_map[current_gy, current_gx]
                next_height = self.terrain_map[next_gy, next_gx]
                
                if abs(next_height - current_height) <= 1:
                    enemy['pos'] = next_pos
                else: 
                    perp_dir = np.array([-direction[1], direction[0]])
                    slide_pos = enemy['pos'] + perp_dir * enemy['speed']
                    slide_gx = int(slide_pos[0] / self.TERRAIN_GRID_SIZE)
                    slide_gy = int(slide_pos[1] / self.TERRAIN_GRID_SIZE)
                    if 0 <= slide_gx < self.GRID_W and 0 <= slide_gy < self.GRID_H:
                        slide_height = self.terrain_map[slide_gy, slide_gx]
                        if abs(slide_height - current_height) <= 1:
                            enemy['pos'] = slide_pos

    def _update_projectiles(self):
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _handle_collisions(self):
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < enemy['size']:
                    self._create_particles(proj['pos'], 10, self.COLOR_ENEMY_A, 2)
                    enemy['health'] -= proj['damage']
                    self.reward_this_step += 0.1
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    
                    if enemy['health'] <= 0:
                        self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY_B, 4)
                        if enemy in self.enemies: self.enemies.remove(enemy)
                        self.reward_this_step += 1.0
                        self.score += 100
                    break

        for enemy in self.enemies[:]:
            for tower in self.towers:
                if tower['destroyed']: continue
                if np.linalg.norm(enemy['pos'] - tower['pos']) < tower['base_size'] + enemy['size']:
                    tower['health'] -= 1
                    self.reward_this_step -= 0.01
                    self._create_particles(enemy['pos'], 5, (200,200,200), 1)
                    if enemy in self.enemies: self.enemies.remove(enemy)
                    
                    if tower['health'] <= 0 and not tower['destroyed']:
                        tower['destroyed'] = True
                        self._create_particles(tower['pos'], 50, self.COLOR_TOWER, 5)
                        self.reward_this_step -= 1.0
                    break
        
        self.projectiles = [p for p in self.projectiles if p['lifespan'] > 0 and 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        self.enemies = [e for e in self.enemies if e['lifespan'] > 0]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_termination(self):
        num_active_towers = len([t for t in self.towers if not t['destroyed']])
        if num_active_towers == 0:
            self.game_over = True
            self.reward_this_step -= 100
            return True
        
        if self.current_wave > self.NUM_WAVES and not self.enemies:
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
        self._render_terrain()
        self._render_particles()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()

    def _render_terrain(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                height = self.terrain_map[y, x]
                if height > 0:
                    color = self.COLOR_TERRAIN_LOW.lerp(self.COLOR_TERRAIN_HIGH, height / 5.0)
                    rect = pygame.Rect(x * self.TERRAIN_GRID_SIZE, y * self.TERRAIN_GRID_SIZE,
                                       self.TERRAIN_GRID_SIZE, self.TERRAIN_GRID_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
        for x in range(0, self.WIDTH, self.TERRAIN_GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TERRAIN_GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_towers(self):
        for i, tower in enumerate(self.towers):
            if tower['destroyed']:
                pygame.draw.circle(self.screen, (50,50,60), (int(tower['pos'][0]), int(tower['pos'][1])), tower['base_size'])
                continue

            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            
            if i == self.selected_tower_index:
                s = pygame.Surface((100, 100), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_TOWER_SELECTED, 50), (50, 50), 50)
                pygame.draw.circle(s, (*self.COLOR_TOWER_SELECTED, 80), (50, 50), 45)
                self.screen.blit(s, (pos[0]-50, pos[1]-50))

            pygame.draw.circle(self.screen, self.COLOR_TOWER, pos, tower['base_size'])
            pygame.draw.circle(self.screen, (0,0,0), pos, tower['base_size'] - 2)
            
            angle = tower['angle']
            height = tower['height']
            cannon_end = (pos[0] + math.cos(angle) * height, pos[1] + math.sin(angle) * height)
            pygame.draw.line(self.screen, self.COLOR_TOWER, pos, cannon_end, 8)
            pygame.draw.circle(self.screen, self.COLOR_TOWER, (int(cannon_end[0]), int(cannon_end[1])), 6)

            health_pct = tower['health'] / tower['max_health']
            bar_width = 40
            bar_height = 5
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - tower['base_size'] - 15
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = int(enemy['size'])
            color = self.COLOR_ENEMY_A if enemy['type'] == 'A' else self.COLOR_ENEMY_B
            
            if enemy['type'] == 'A':
                points = [
                    (pos[0], pos[1] - size),
                    (pos[0] - size * 0.866, pos[1] + size * 0.5),
                    (pos[0] + size * 0.866, pos[1] + size * 0.5)
                ]
            else:
                points = []
                for i in range(5):
                    angle = i * (2 * math.pi / 5) + math.pi/2
                    points.append((pos[0] + size * math.cos(angle), pos[1] - size * math.sin(angle)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_projectiles(self):
        for p in self.projectiles:
            start = (int(p['pos'][0]), int(p['pos'][1]))
            end = (int(p['pos'][0] - p['vel'][0] * 2), int(p['pos'][1] - p['vel'][1] * 2))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start, end, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        wave_str = f"WAVE: {self.current_wave}/{self.NUM_WAVES}"
        if self.wave_timer > 0 and self.current_wave == 0:
            wave_str = f"FIRST WAVE IN: {self.wave_timer // 30 + 1}"
        elif self.wave_timer > 0:
            wave_str = f"NEXT WAVE IN: {self.wave_timer // 30 + 1}"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        shift_held = self.last_action[2] == 1
        mode_str = "MODE: TERRAFORM (SHIFT)" if shift_held else "MODE: TOWER CONTROL"
        mode_text = self.font_small.render(mode_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(mode_text, (10, self.HEIGHT - mode_text.get_height() - 10))
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            end_text_str = "VICTORY!" if self.current_wave > self.NUM_WAVES else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave}

    def _create_tower(self, x, y):
        return {
            'pos': np.array([x, y], dtype=float),
            'height': 20, 'angle': -math.pi/2, 'base_size': 15,
            'health': 500, 'max_health': 500,
            'fire_cooldown': 0, 'destroyed': False
        }

    def _create_projectile(self, tower):
        speed = 8
        vel = np.array([math.cos(tower['angle']), math.sin(tower['angle'])]) * speed
        start_pos = tower['pos'] + vel * (tower['height'] / speed)
        return {
            'pos': start_pos, 'vel': vel,
            'damage': 10 + tower['height'] * 0.5, 'lifespan': 80
        }

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense Gym Environment")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        clock.tick(30)

    pygame.quit()