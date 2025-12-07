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
        "Controls: Arrow keys to move the placement cursor. Space to build a "
        "Gatling Tower. Shift to build a Cannon Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Survive 10 waves of enemies by "
        "strategically placing towers to protect your base."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 80)
    COLOR_PATH_BORDER = (50, 65, 100)
    COLOR_BASE = (0, 150, 100)
    COLOR_BASE_GLOW = (0, 200, 150)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_GLOW = (255, 100, 100)
    COLOR_TOWER_1 = (0, 150, 255) # Gatling
    COLOR_TOWER_2 = (255, 150, 0) # Cannon
    COLOR_PROJECTILE_1 = (100, 200, 255)
    COLOR_PROJECTILE_2 = (255, 200, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (200, 0, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR_FG = (100, 220, 100)
    
    # Game parameters
    MAX_WAVES = 10
    BASE_MAX_HEALTH = 100
    STARTING_GOLD = 250
    GOLD_PER_KILL = 15
    WAVE_CLEAR_BONUS = 50
    
    # Tower stats
    TOWER_SPECS = {
        1: {'cost': 75, 'range': 80, 'damage': 0.5, 'fire_rate': 5, 'projectile_speed': 8, 'color': COLOR_TOWER_1, 'proj_color': COLOR_PROJECTILE_1}, # Gatling
        2: {'cost': 125, 'range': 150, 'damage': 5, 'fire_rate': 45, 'projectile_speed': 10, 'color': COLOR_TOWER_2, 'proj_color': COLOR_PROJECTILE_2} # Cannon
    }

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 50)
        
        # Path and grid setup
        self.path_waypoints = [
            (-20, 200), (100, 200), (100, 100), (450, 100),
            (450, 300), (250, 300), (250, 350), (self.SCREEN_WIDTH + 20, 350)
        ]
        self.tower_grid = self._create_tower_grid()
        
        # Initialize state variables
        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.BASE_MAX_HEALTH
        self.gold = self.STARTING_GOLD
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        # Reset tower grid occupancy
        if hasattr(self, 'tower_grid'):
            for col in self.tower_grid:
                for tile in col:
                    tile['occupied'] = False

        self.cursor_grid_pos = [len(self.tower_grid) // 2, len(self.tower_grid[0]) // 2]
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_in_progress = False
        self.wave_spawn_timer = 0
        self.wave_enemies_to_spawn = []
        self.time_to_next_wave = 150 # 5 seconds at 30fps

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Place Towers ---
        if space_held and not self.last_space_held:
            if self._place_tower(1):
                pass # sfx: place_gatling
        if shift_held and not self.last_shift_held:
            if self._place_tower(2):
                pass # sfx: place_cannon
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Game Logic Updates ---
        self._update_towers()
        reward += self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        reward += self._update_wave_manager()
        
        self.steps += 1
        terminated = self.game_over
        
        # --- Terminal Rewards ---
        if terminated and not self.game_won:
            reward = -100
        elif terminated and self.game_won:
            reward = 100

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_grid_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1 # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1 # Right
        
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, len(self.tower_grid) - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, len(self.tower_grid[0]) - 1)

    def _create_tower_grid(self):
        grid = []
        grid_x_points = range(50, self.SCREEN_WIDTH - 50, 70)
        grid_y_points = range(50, self.SCREEN_HEIGHT - 50, 70)

        for x in grid_x_points:
            col = []
            for y in grid_y_points:
                pos = (x, y)
                on_path = False
                for i in range(len(self.path_waypoints) - 1):
                    p1 = pygame.Vector2(self.path_waypoints[i])
                    p2 = pygame.Vector2(self.path_waypoints[i+1])
                    point = pygame.Vector2(pos)
                    
                    # Check distance to line segment
                    l2 = p1.distance_squared_to(p2)
                    if l2 == 0.0:
                        dist = point.distance_to(p1)
                    else:
                        t = max(0, min(1, (point - p1).dot(p2 - p1) / l2))
                        projection = p1 + t * (p2 - p1)
                        dist = point.distance_to(projection)
                    
                    if dist < 40: # Path width + buffer
                        on_path = True
                        break
                col.append({'pos': pos, 'occupied': False, 'buildable': not on_path})
            grid.append(col)
        return grid

    def _place_tower(self, tower_type):
        spec = self.TOWER_SPECS[tower_type]
        if self.gold < spec['cost']:
            # sfx: error_buzz
            return False
            
        grid_x, grid_y = self.cursor_grid_pos
        tile = self.tower_grid[grid_x][grid_y]
        
        if tile['occupied'] or not tile['buildable']:
            # sfx: error_buzz
            return False
        
        self.gold -= spec['cost']
        tile['occupied'] = True
        
        new_tower = {
            'pos': pygame.Vector2(tile['pos']),
            'type': tower_type,
            'cooldown': 0,
            **spec
        }
        self.towers.append(new_tower)
        return True

    def _update_wave_manager(self):
        if self.game_over: return 0
        
        if not self.wave_in_progress:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0:
                self._start_next_wave()
        else:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0 and self.wave_enemies_to_spawn:
                self.enemies.append(self.wave_enemies_to_spawn.pop(0))
                self.wave_spawn_timer = 20 # Time between spawns
            
            if not self.wave_enemies_to_spawn and not self.enemies:
                self.wave_in_progress = False
                self.gold += self.WAVE_CLEAR_BONUS
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
                    self.game_won = True
                else:
                    self.time_to_next_wave = 300 # 10 seconds
                return 1.0 # Wave clear reward
        return 0

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        num_enemies = 5 + self.wave_number
        enemy_health = 10 * (1.1 ** (self.wave_number - 1))
        
        for _ in range(num_enemies):
            enemy = {
                'pos': pygame.Vector2(self.path_waypoints[0]),
                'health': enemy_health,
                'max_health': enemy_health,
                'waypoint_index': 1,
                'speed': self.np_random.uniform(1.0, 1.5)
            }
            self.wave_enemies_to_spawn.append(enemy)

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = tower['pos'].distance_to(enemy['pos'])
                    if dist <= tower['range'] and dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    # sfx: fire_tower
                    self.projectiles.append({
                        'pos': tower['pos'].copy(),
                        'target': target,
                        'speed': tower['projectile_speed'],
                        'damage': tower['damage'],
                        'color': tower['proj_color']
                    })
                    tower['cooldown'] = tower['fire_rate']

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['waypoint_index'] >= len(self.path_waypoints):
                self.base_health -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage
                if self.base_health <= 0:
                    self.game_over = True
                continue
            
            target_pos = pygame.Vector2(self.path_waypoints[enemy['waypoint_index']])
            direction = (target_pos - enemy['pos'])
            
            if direction.length() < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['waypoint_index'] += 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']
            
            if enemy['health'] <= 0:
                self.gold += self.GOLD_PER_KILL
                reward += 0.1
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 20)
                # sfx: enemy_death
        return reward
        
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if not proj['target'] in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            direction = (target_pos - proj['pos'])
            
            if direction.length() < proj['speed']:
                proj['target']['health'] -= proj['damage']
                self.projectiles.remove(proj)
                self._create_particles(target_pos, proj['color'], 5, size=2)
                # sfx: projectile_hit
            else:
                proj['pos'] += direction.normalize() * proj['speed']

    def _create_particles(self, pos, color, count, size=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(size-1, size)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

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
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number
        }
    
    def _render_game(self):
        # Draw path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, 34)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 30)

        # Draw base
        base_pos = self.path_waypoints[-1]
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), 20, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), 18, self.COLOR_BASE)
        
        # Draw tower placement grid
        for col in self.tower_grid:
            for tile in col:
                if tile['buildable']:
                    pygame.gfxdraw.aacircle(self.screen, int(tile['pos'][0]), int(tile['pos'][1]), 10, (60, 70, 100))

        # Draw towers
        for tower in self.towers:
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            pygame.draw.rect(self.screen, tower['color'], (pos[0]-10, pos[1]-10, 20, 20))
            pygame.draw.rect(self.screen, (255,255,255), (pos[0]-10, pos[1]-10, 20, 20), 1)

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, proj['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, (255,255,255))
            
        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0]-10, pos[1]-15, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (pos[0]-10, pos[1]-15, 20 * health_pct, 3))
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'] * (p['lifetime'] / 30.0))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

        # Draw cursor
        grid_x, grid_y = self.cursor_grid_pos
        tile = self.tower_grid[grid_x][grid_y]
        cursor_pos = tile['pos']
        is_valid = not tile['occupied'] and tile['buildable']
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        pygame.draw.rect(self.screen, cursor_color, (cursor_pos[0]-15, cursor_pos[1]-15, 30, 30), 2)

    def _render_ui(self):
        # Top Bar
        pygame.draw.rect(self.screen, (10, 12, 20), (0, 0, self.SCREEN_WIDTH, 30))
        
        # Gold
        gold_text = self.font_ui.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 7))
        
        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and not self.game_over:
            next_wave_secs = self.time_to_next_wave // 30
            wave_str = f"NEXT WAVE IN: {next_wave_secs}"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH // 2 - wave_text.get_width() // 2, 7))
        
        # Base Health
        health_text = self.font_ui.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - 220, 7))
        health_pct = max(0, self.base_health / self.BASE_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.SCREEN_WIDTH - 120, 7, 110, 16))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (self.SCREEN_WIDTH - 119, 8, 108 * health_pct, 14))

        # Game Over / Victory Message
        if self.game_over:
            msg_str = "VICTORY" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            msg_text = self.font_msg.render(msg_str, True, color)
            pos = (self.SCREEN_WIDTH // 2 - msg_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_text.get_height() // 2)
            self.screen.blit(msg_text, pos)

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
        
        # print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The main block is for human play and visualization, which requires a display.
    # It will not run in a strictly headless environment like the validation server.
    # To run this, you might need to comment out the SDL_VIDEODRIVER line.
    try:
        del os.environ['SDL_VIDEODRIVER']
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    print(env.game_description)

    # Game loop for human play
    running = True
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # No-op
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
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()