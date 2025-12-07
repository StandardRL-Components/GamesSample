
# Generated: 2025-08-28T03:54:51.885398
# Source Brief: brief_05078.md
# Brief Index: 5078

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Place towers to defend your base "
        "from waves of enemies marching along a path. Survive all 10 waves to win."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (25, 35, 45)
        self.COLOR_BASE = (0, 180, 120)
        self.COLOR_ENEMY = (210, 50, 50)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 150)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 150)

        # Tower definitions
        self.TOWER_SPECS = {
            "machine_gun": {
                "cost": 100, "range": 80, "fire_rate": 0.2, "damage": 5, 
                "color": (0, 150, 255), "projectile_speed": 10
            },
            "cannon": {
                "cost": 250, "range": 120, "fire_rate": 1.5, "damage": 50,
                "color": (255, 150, 0), "projectile_speed": 7
            }
        }
        self.tower_types = list(self.TOWER_SPECS.keys())

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = None
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback for older gym versions or if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.step_rewards = []
        
        # Player state
        self.money = 250
        self.base_health = 100
        self.cursor_pos = [self.GRID_W // 4, self.GRID_H // 2]
        self.selected_tower_idx = 0
        
        # Input state
        self.last_space_held = False
        self.last_shift_held = False

        # Wave management
        self.current_wave = 0
        self.wave_timer = 5 * self.FPS # Time until first wave
        self.time_between_waves = 10 * self.FPS
        self.enemies_to_spawn = []
        
        # Game entities
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Level generation
        self._generate_path()
        
        return self._get_observation(), self._get_info()

    def _generate_path(self):
        self.path_waypoints = []
        self.path_cells = set()
        
        y = self.rng.integers(self.GRID_H // 4, self.GRID_H * 3 // 4)
        x = 0
        self.path_waypoints.append( (x * self.GRID_SIZE + self.GRID_SIZE//2, y * self.GRID_SIZE + self.GRID_SIZE//2) )
        self.path_cells.add((x, y))

        while x < self.GRID_W - 2:
            move_y = self.rng.integers(-2, 3)
            move_x = self.rng.integers(1, 4)
            
            next_y = np.clip(y + move_y, 1, self.GRID_H - 2)
            next_x = np.clip(x + move_x, 0, self.GRID_W - 2)

            for i in range(x, next_x + 1): self.path_cells.add((i, y))
            for i in range(min(y, next_y), max(y, next_y) + 1): self.path_cells.add((next_x, i))
            
            x, y = next_x, next_y
            self.path_waypoints.append( (x * self.GRID_SIZE + self.GRID_SIZE//2, y * self.GRID_SIZE + self.GRID_SIZE//2) )

        self.base_pos_grid = (self.GRID_W - 1, y)
        self.base_pos_px = ((self.GRID_W - 1) * self.GRID_SIZE, y * self.GRID_SIZE)
        self.path_waypoints.append((self.base_pos_px[0], self.base_pos_px[1] + self.GRID_SIZE//2))
        self.path_cells.add(self.base_pos_grid)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_rewards = []
        
        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        reward = sum(self.step_rewards)
        terminated = self._check_termination()

        if terminated:
            if self.base_health <= 0:
                reward -= 100 # Loss penalty
            elif self.current_wave > 10:
                reward += 100 # Win bonus
        
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
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
        
        # Place tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _is_placement_valid(self):
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_idx]]
        if self.money < spec['cost']: return False
        if tuple(self.cursor_pos) in self.path_cells: return False
        for tower in self.towers:
            if tower['grid_pos'] == self.cursor_pos: return False
        return True

    def _place_tower(self):
        if not self._is_placement_valid():
            # sfx: error_buzz
            return

        spec_name = self.tower_types[self.selected_tower_idx]
        spec = self.TOWER_SPECS[spec_name]
        self.money -= spec['cost']

        px_pos = (
            self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
            self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        )
        
        new_tower = {
            'spec_name': spec_name,
            'grid_pos': list(self.cursor_pos),
            'px_pos': px_pos,
            'fire_cooldown': 0,
            'target': None
        }
        self.towers.append(new_tower)
        # sfx: place_tower

    def _update_game_state(self):
        self._update_waves()
        self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        
    def _update_waves(self):
        if self.enemies or self.enemies_to_spawn:
            return
        
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.current_wave += 1
            if self.current_wave > 10:
                return # Game won
            
            self.step_rewards.append(1.0) # Wave survival bonus
            self.wave_timer = self.time_between_waves
            
            num_enemies = 5 + self.current_wave * 2
            enemy_health = 50 + (self.current_wave - 1) * 20
            enemy_speed = 0.8 + (self.current_wave - 1) * 0.1
            enemy_value = 10 + (self.current_wave - 1) * 2

            for i in range(num_enemies):
                self.enemies_to_spawn.append({
                    'health': enemy_health,
                    'max_health': enemy_health,
                    'speed': enemy_speed,
                    'value': enemy_value,
                    'spawn_delay': i * (self.FPS // 2) # Stagger spawns
                })
    
    def _update_enemies(self):
        # Spawn enemies from the queue
        if self.enemies_to_spawn:
            self.enemies_to_spawn[0]['spawn_delay'] -= 1
            if self.enemies_to_spawn[0]['spawn_delay'] <= 0:
                spawn = self.enemies_to_spawn.pop(0)
                new_enemy = {
                    'pos': np.array(self.path_waypoints[0], dtype=float),
                    'health': spawn['health'],
                    'max_health': spawn['max_health'],
                    'speed': spawn['speed'],
                    'value': spawn['value'],
                    'waypoint_idx': 1,
                    'dist_traveled': 0
                }
                self.enemies.append(new_enemy)

        # Move existing enemies
        for enemy in reversed(self.enemies):
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            target_pos = np.array(self.path_waypoints[enemy['waypoint_idx']])
            direction = target_pos - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist < enemy['speed']:
                enemy['waypoint_idx'] += 1
                enemy['dist_traveled'] += dist
            else:
                move_vec = (direction / dist) * enemy['speed']
                enemy['pos'] += move_vec
                enemy['dist_traveled'] += enemy['speed']
                
    def _update_towers(self):
        for tower in self.towers:
            tower['fire_cooldown'] = max(0, tower['fire_cooldown'] - 1/self.FPS)
            if tower['fire_cooldown'] > 0:
                continue
            
            spec = self.TOWER_SPECS[tower['spec_name']]
            
            # Find best target (furthest traveled enemy in range)
            best_target = None
            max_dist_traveled = -1
            
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower['px_pos']) - enemy['pos'])
                if dist <= spec['range'] and enemy['dist_traveled'] > max_dist_traveled:
                    max_dist_traveled = enemy['dist_traveled']
                    best_target = enemy
            
            if best_target:
                tower['fire_cooldown'] = spec['fire_rate']
                
                # Predict target position to lead the shot
                dist_to_target = np.linalg.norm(np.array(tower['px_pos']) - best_target['pos'])
                time_to_impact = dist_to_target / spec['projectile_speed']
                
                # Simplified prediction: assume straight path from current pos
                if best_target['waypoint_idx'] < len(self.path_waypoints):
                    target_waypoint = np.array(self.path_waypoints[best_target['waypoint_idx']])
                    direction = target_waypoint - best_target['pos']
                    dist_to_waypoint = np.linalg.norm(direction)
                    if dist_to_waypoint > 1:
                        direction /= dist_to_waypoint
                        predicted_pos = best_target['pos'] + direction * best_target['speed'] * time_to_impact
                    else:
                        predicted_pos = best_target['pos']
                else:
                    predicted_pos = best_target['pos']
                
                self.projectiles.append({
                    'pos': np.array(tower['px_pos'], dtype=float),
                    'target_pos': predicted_pos,
                    'damage': spec['damage'],
                    'speed': spec['projectile_speed'],
                    'color': spec['color']
                })
                # sfx: shoot_laser
                
    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            direction = proj['target_pos'] - proj['pos']
            dist = np.linalg.norm(direction)

            if dist < proj['speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'] += (direction / dist) * proj['speed']
            
            hit = False
            for enemy in self.enemies:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < self.GRID_SIZE / 2:
                    enemy['health'] -= proj['damage']
                    # sfx: enemy_hit
                    self._create_particles(proj['pos'], proj['color'], 5)
                    
                    if enemy['health'] <= 0:
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 2.0)
                        self.money += enemy['value']
                        self.step_rewards.append(0.1)
                        self.enemies.remove(enemy)
                        # sfx: enemy_explode
                    
                    hit = True
                    break
            
            if hit or dist < proj['speed']:
                self.projectiles.remove(proj)

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = (self.rng.random() * 2 + 1) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.rng.integers(10, 20),
                'color': color,
                'radius': self.rng.random() * 2 + 1
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.current_wave > 10 and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            return True
        if self.steps >= 20000: # Max episode length
            self.game_over = True
            return True
        return False

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
            "money": self.money,
            "base_health": self.base_health,
            "wave": self.current_wave,
        }

    def _render_game(self):
        # Grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.GRID_SIZE)
        
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (*self.base_pos_px, self.GRID_SIZE, self.GRID_SIZE))

        # Towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['spec_name']]
            pygame.draw.circle(self.screen, spec['color'], tower['px_pos'], self.GRID_SIZE // 3)
            # Draw range indicator when placing or hovering
            dist_to_cursor = np.linalg.norm(np.array(tower['px_pos']) - np.array(self.cursor_pos)*self.GRID_SIZE)
            if dist_to_cursor < self.GRID_SIZE:
                pygame.gfxdraw.aacircle(self.screen, int(tower['px_pos'][0]), int(tower['px_pos'][1]), spec['range'], (255,255,255,50))


        # Enemies
        for enemy in self.enemies:
            pos = tuple(map(int, enemy['pos']))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, self.GRID_SIZE // 2 - 2)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = self.GRID_SIZE
            bar_h = 4
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - self.GRID_SIZE
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, int(bar_w * health_ratio), bar_h))
            
        # Projectiles
        for proj in self.projectiles:
            pos = tuple(map(int, proj['pos']))
            pygame.draw.circle(self.screen, proj['color'], pos, 3)
            
        # Particles
        for p in self.particles:
            pos = tuple(map(int, p['pos']))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius'] * (p['life']/20)))
        
        # Cursor
        cursor_px = (self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE)
        cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        color = self.COLOR_CURSOR_VALID if self._is_placement_valid() else self.COLOR_CURSOR_INVALID
        cursor_surface.fill(color)
        self.screen.blit(cursor_surface, cursor_px)
        
        # Tower range preview
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_idx]]
        center_px = (cursor_px[0] + self.GRID_SIZE//2, cursor_px[1] + self.GRID_SIZE//2)
        pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], spec['range'], (255,255,255,50))


    def _render_ui(self):
        # Top bar background
        top_bar_rect = pygame.Rect(0, 0, self.WIDTH, 30)
        pygame.draw.rect(self.screen, (0,0,0,150), top_bar_rect)

        # Money
        money_text = self.font_small.render(f"$ {self.money}", True, (255, 223, 0))
        self.screen.blit(money_text, (10, 7))

        # Base Health
        health_text = self.font_small.render(f"Base: {self.base_health}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (120, 7))
        
        # Wave info
        wave_str = f"Wave: {self.current_wave}/10"
        if self.wave_timer > 0 and not self.enemies and not self.enemies_to_spawn and self.current_wave < 10:
            wave_str += f" (Next in {self.wave_timer/self.FPS:.0f}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (250, 7))

        # Selected Tower Info (near cursor)
        spec_name = self.tower_types[self.selected_tower_idx]
        spec = self.TOWER_SPECS[spec_name]
        
        info_x = self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE + 5
        info_y = self.cursor_pos[1] * self.GRID_SIZE
        if info_x > self.WIDTH - 150: info_x -= (160 + self.GRID_SIZE)

        name_text = self.font_small.render(f"{spec_name.replace('_', ' ').title()}", True, self.COLOR_TEXT)
        cost_color = (255, 223, 0) if self.money >= spec['cost'] else (255, 80, 80)
        cost_text = self.font_small.render(f"Cost: ${spec['cost']}", True, cost_color)
        
        self.screen.blit(name_text, (info_x, info_y))
        self.screen.blit(cost_text, (info_x, info_y + 18))

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "VICTORY!" if self.base_health > 0 else "DEFEAT"
            color = (0, 255, 100) if self.base_health > 0 else (255, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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

# Example usage to run and visualize the game
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- Pygame setup for visualization ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard to Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Money: {info['money']}")

        if terminated or truncated:
            print("Game Over!")
            print(f"Final Score: {info['score']:.2f}, Final Wave: {info['wave']}")
            obs, info = env.reset(seed=random.randint(0, 10000))

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()