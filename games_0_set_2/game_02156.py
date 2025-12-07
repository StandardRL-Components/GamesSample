
# Generated: 2025-08-27T19:26:30.692877
# Source Brief: brief_02156.md
# Brief Index: 2156

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (45, 48, 56)
    COLOR_PATH_BORDER = (65, 68, 76)
    COLOR_GRID = (35, 38, 46)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BASE = (80, 200, 120)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_HEALTH_BG = (10, 10, 10)
    COLOR_HEALTH_FG = (100, 255, 100)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    TOWER_SPECS = [
        {'name': 'Gatling', 'cost': 50, 'range': 80, 'damage': 4, 'cooldown': 10, 'color': (0, 150, 255), 'projectile_speed': 8, 'splash': 0},
        {'name': 'Cannon', 'cost': 100, 'range': 120, 'damage': 15, 'cooldown': 45, 'color': (255, 200, 0), 'projectile_speed': 6, 'splash': 0},
        {'name': 'Artillery', 'cost': 150, 'range': 200, 'damage': 10, 'cooldown': 90, 'color': (200, 80, 255), 'projectile_speed': 4, 'splash': 30},
    ]

    PATH_WAYPOINTS = [
        (-20, 200), (100, 200), (100, 100), (300, 100), (300, 300),
        (540, 300), (540, 200), (SCREEN_WIDTH + 20, 200)
    ]
    
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
        self.font_small = pygame.font.SysFont("sans-serif", 16)
        self.font_large = pygame.font.SysFont("sans-serif", 48)

        self._init_path_grid()
        self.reset()
        
        # Self-check
        # self.validate_implementation()

    def _init_path_grid(self):
        self.path_rects = []
        for i in range(len(self.PATH_WAYPOINTS) - 1):
            p1 = self.PATH_WAYPOINTS[i]
            p2 = self.PATH_WAYPOINTS[i+1]
            rect = pygame.Rect(min(p1[0], p2[0]) - 15, min(p1[1], p2[1]) - 15,
                               abs(p1[0] - p2[0]) + 30, abs(p1[1] - p2[1]) + 30)
            self.path_rects.append(rect)

        self.buildable_grid = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                cell_rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                on_path = any(cell_rect.colliderect(path_rect) for path_rect in self.path_rects)
                if not on_path:
                    self.buildable_grid.add((x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = 100
        self.resources = 100
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.enemy_spawn_timer = 0
        self.base_spawn_rate = 120
        self.min_spawn_rate = 30
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.last_shift_state = 0
        self.last_space_state = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        placement_cost = self._handle_actions(action)
        reward += placement_cost

        step_rewards = self._update_game_state()
        reward += step_rewards
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100
            else:
                reward -= 100
            self.score += reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if shift_held and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle_sound

        placement_cost = 0
        if space_held and not self.last_space_state:
            placement_cost = self._place_tower()
            # sfx: place_tower_sound or error_sound

        self.last_space_state = space_held
        self.last_shift_state = shift_held
        return placement_cost

    def _place_tower(self):
        grid_pos = tuple(self.cursor_pos)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        is_buildable = grid_pos in self.buildable_grid
        is_occupied = any(t['grid_pos'] == grid_pos for t in self.towers)
        has_resources = self.resources >= spec['cost']

        if is_buildable and not is_occupied and has_resources:
            self.resources -= spec['cost']
            world_pos = [(grid_pos[0] + 0.5) * self.GRID_SIZE, (grid_pos[1] + 0.5) * self.GRID_SIZE]
            
            new_tower = {
                'grid_pos': grid_pos,
                'pos': world_pos,
                'type': self.selected_tower_type,
                'fire_timer': 0,
                'target': None
            }
            self.towers.append(new_tower)
            self._create_particles(world_pos, 20, spec['color'], 1, 3)
            return -spec['cost'] / 100.0 # Small negative reward for spending
        return 0

    def _update_game_state(self):
        reward = 0
        
        self._spawn_enemies()
        reward += self._update_towers()
        reward += self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        return reward

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            speed_multiplier = 1.0 + (self.steps / 200) * 0.1
            
            # Spawn stronger enemies over time
            enemy_type_chance = self.np_random.random()
            if enemy_type_chance > 0.9 and self.steps > 500:
                health, speed, size, value = 100, 0.7, 12, 5
            elif enemy_type_chance > 0.7 and self.steps > 250:
                health, speed, size, value = 40, 1.0, 9, 2
            else:
                health, speed, size, value = 20, 1.2, 7, 1

            new_enemy = {
                'pos': list(self.PATH_WAYPOINTS[0]),
                'path_index': 1,
                'health': health,
                'max_health': health,
                'speed': speed * speed_multiplier,
                'size': size,
                'value': value
            }
            self.enemies.append(new_enemy)

            spawn_rate_progress = self.steps / self.MAX_STEPS
            current_spawn_rate = self.base_spawn_rate + (self.min_spawn_rate - self.base_spawn_rate) * spawn_rate_progress
            self.enemy_spawn_timer = int(current_spawn_rate)

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            tower['fire_timer'] = max(0, tower['fire_timer'] - 1)
            
            if tower['fire_timer'] == 0:
                # Find a new target
                target = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                    if dist <= spec['range'] and dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                tower['target'] = target

                if tower['target']:
                    # sfx: tower_fire_sound
                    proj = {
                        'pos': list(tower['pos']),
                        'target': tower['target'],
                        'spec': spec
                    }
                    self.projectiles.append(proj)
                    tower['fire_timer'] = spec['cooldown']
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            spec = proj['spec']
            target = proj['target']

            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = [target['pos'][0] - proj['pos'][0], target['pos'][1] - proj['pos'][1]]
            dist = math.hypot(*direction)
            
            if dist < spec['projectile_speed']:
                # Hit target
                reward += self._damage_enemy(target, spec['damage'])
                
                # Splash damage
                if spec['splash'] > 0:
                    for enemy in self.enemies:
                        if enemy is not target:
                            splash_dist = math.hypot(target['pos'][0] - enemy['pos'][0], target['pos'][1] - enemy['pos'][1])
                            if splash_dist <= spec['splash']:
                                reward += self._damage_enemy(enemy, spec['damage'] / 2) # Half damage for splash
                
                self._create_particles(proj['pos'], 10, spec['color'], 0.5, 2)
                self.projectiles.remove(proj)
                # sfx: projectile_hit_sound
            else:
                # Move projectile
                norm_dir = [d / dist for d in direction]
                proj['pos'][0] += norm_dir[0] * spec['projectile_speed']
                proj['pos'][1] += norm_dir[1] * spec['projectile_speed']
        return reward

    def _damage_enemy(self, enemy, damage):
        enemy['health'] -= damage
        return 0.1 # Reward for any hit

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.score += enemy['value']
                self.resources += enemy['value'] * 10
                self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY, 1, 4)
                self.enemies.remove(enemy)
                # sfx: enemy_death_sound
                continue

            target_waypoint = self.PATH_WAYPOINTS[enemy['path_index']]
            direction = [target_waypoint[0] - enemy['pos'][0], target_waypoint[1] - enemy['pos'][1]]
            dist = math.hypot(*direction)

            if dist < enemy['speed']:
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.PATH_WAYPOINTS):
                    self.base_health -= 10
                    self.enemies.remove(enemy)
                    # sfx: base_damage_sound
                    continue
            else:
                norm_dir = [d / dist for d in direction]
                enemy['pos'][0] += norm_dir[0] * enemy['speed']
                enemy['pos'][1] += norm_dir[1] * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_scale, life_scale):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * speed_scale + 0.5
            life = int(self.np_random.random() * 10 * life_scale + 10)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': life,
                'max_life': life,
            })

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            self.victory = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.PATH_WAYPOINTS, 34)
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 30)

        # Base
        base_pos = (self.SCREEN_WIDTH - 20, 180)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (*base_pos, 20, 40))

        # Buildable Grid (subtle)
        for x, y in self.buildable_grid:
            pygame.draw.rect(self.screen, self.COLOR_GRID, (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), 1)

        # Towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            x, y = int(tower['pos'][0]), int(tower['pos'][1])
            
            # Pulsing effect for readiness
            if tower['fire_timer'] == 0:
                pulse_rad = (self.steps % 15) + 5
                pulse_alpha = 100 - (pulse_rad * 5)
                pygame.gfxdraw.filled_circle(self.screen, x, y, pulse_rad, (*spec['color'], pulse_alpha))

            # Tower shape
            p1 = (x, y - 10)
            p2 = (x - 8, y + 6)
            p3 = (x + 8, y + 6)
            pygame.gfxdraw.aatrigon(self.screen, *p1, *p2, *p3, spec['color'])
            pygame.gfxdraw.filled_trigon(self.screen, *p1, *p2, *p3, spec['color'])

        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            size = int(enemy['size'])
            pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = size * 2
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x - bar_width//2, y - size - 8, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (x - bar_width//2, y - size - 8, int(bar_width * health_ratio), 5))

        # Projectiles
        for proj in self.projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.draw.rect(self.screen, proj['spec']['color'], (x - 2, y - 2, 4, 4))
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos, 2, color)

        # Cursor
        cursor_world_x = self.cursor_pos[0] * self.GRID_SIZE
        cursor_world_y = self.cursor_pos[1] * self.GRID_SIZE
        cursor_rect = pygame.Rect(cursor_world_x, cursor_world_y, self.GRID_SIZE, self.GRID_SIZE)
        
        is_buildable = tuple(self.cursor_pos) in self.buildable_grid
        is_occupied = any(t['grid_pos'] == tuple(self.cursor_pos) for t in self.towers)
        can_afford = self.resources >= self.TOWER_SPECS[self.selected_tower_type]['cost']
        is_valid = is_buildable and not is_occupied and can_afford
        
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        # Use a surface for transparency
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, (cursor_world_x, cursor_world_y))
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 1)

    def _render_ui(self):
        # Base Health Bar
        health_ratio = max(0, self.base_health / 100)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.SCREEN_WIDTH // 2 - bar_width // 2, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.SCREEN_WIDTH // 2 - bar_width // 2, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"BASE HEALTH: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH // 2 - health_text.get_width() // 2, 12))

        # Resources
        res_text = self.font_small.render(f"RESOURCES: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 10))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Selected Tower UI
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_small.render(f"Tower: {spec['name']} | Cost: ${spec['cost']}", True, spec['color'])
        self.screen.blit(tower_text, (10, self.SCREEN_HEIGHT - 25))

        # Game Over / Victory
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "VICTORY" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    env.validate_implementation()
    
    # Example of a random agent
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    print(f"Episode finished after {step_count} steps.")
    print(f"Total reward: {total_reward}")
    print(f"Final info: {info}")

    env.close()