
# Generated: 2025-08-28T04:00:51.716546
# Source Brief: brief_05118.md
# Brief Index: 5118

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, size, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.color = color
        self.lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

class Enemy:
    def __init__(self, health, speed, path_waypoints, value):
        self.pos = pygame.math.Vector2(path_waypoints[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path_waypoints = path_waypoints
        self.waypoint_index = 1
        self.value = value # Resources gained on destruction
        self.radius = 10

    def update(self):
        if self.waypoint_index < len(self.path_waypoints):
            target_pos = pygame.math.Vector2(self.path_waypoints[self.waypoint_index])
            direction = (target_pos - self.pos)
            if direction.length() < self.speed:
                self.pos = target_pos
                self.waypoint_index += 1
            else:
                self.pos += direction.normalize() * self.speed
        return self.waypoint_index >= len(self.path_waypoints) # Reached end

class Tower:
    def __init__(self, pos, tower_type_def):
        self.pos = pygame.math.Vector2(pos)
        self.type_def = tower_type_def
        self.range = tower_type_def['range']
        self.fire_rate = tower_type_def['fire_rate']
        self.damage = tower_type_def['damage']
        self.color = tower_type_def['color']
        self.cooldown = 0
        self.target = None

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1

        # Find new target if current is dead or out of range
        if self.target and (self.target.health <= 0 or self.pos.distance_to(self.target.pos) > self.range):
            self.target = None
        
        if not self.target:
            closest_enemy = None
            min_dist = float('inf')
            for enemy in enemies:
                dist = self.pos.distance_to(enemy.pos)
                if dist <= self.range and dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
            self.target = closest_enemy
    
    def can_fire(self):
        return self.cooldown == 0 and self.target is not None

class Projectile:
    def __init__(self, start_pos, target, damage, speed=15):
        self.pos = pygame.math.Vector2(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed

    def update(self):
        if self.target.health <= 0:
            return True # Target is already dead, projectile should be removed

        direction = (self.target.pos - self.pos)
        dist = direction.length()

        if dist < self.speed:
            # Hit
            self.target.health -= self.damage
            return True
        else:
            self.pos += direction.normalize() * self.speed
            return False # Still in flight

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle tower types. Press Space to build the selected tower."
    )

    game_description = (
        "A minimalist tower defense game. Strategically place towers to defend your base "
        "from waves of incoming enemies. Survive all waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PATH = (40, 40, 60)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_PROJECTILE = (255, 225, 25)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 100, 100)
        
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 50)

        # --- Game Constants ---
        self.MAX_STEPS = 1500 # Increased from 1000 for better playability
        self.TOTAL_WAVES = 5
        self.BASE_MAX_HEALTH = 100
        
        # --- Tower Definitions ---
        self.TOWER_TYPES = [
            {'name': 'Cannon', 'cost': 25, 'range': 100, 'fire_rate': 30, 'damage': 10, 'color': (0, 130, 200)},
            {'name': 'Missile', 'cost': 75, 'range': 180, 'fire_rate': 90, 'damage': 50, 'color': (245, 130, 48)},
        ]

        # --- World Layout ---
        self.GRID_SIZE = 40
        self.GRID_COLS = 12
        self.GRID_ROWS = 8
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_COLS * self.GRID_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.GRID_SIZE) // 2
        
        self.path_waypoints = [
            (0, 50), (150, 50), (150, 300), (490, 300), (490, 150), (self.SCREEN_WIDTH, 150)
        ]
        self.base_pos = (self.SCREEN_WIDTH - 20, 150)

        # Initialize state variables
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.placement_grid = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.BASE_MAX_HEALTH
        self.resources = 100
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.placement_grid = [[True for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.grid_cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_tower_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.wave_number = 0
        self.wave_state = "PRE_WAVE" # PRE_WAVE, SPAWNING, WAITING
        self.wave_timer = 90 # 3 seconds at 30fps
        self.enemies_to_spawn = []

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return

        self.wave_state = "SPAWNING"
        num_enemies = 3 + (self.wave_number - 1) * 2
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.2
        enemy_health = 20 + (self.wave_number - 1) * 10
        enemy_value = 5 + self.wave_number
        
        self.enemies_to_spawn = [
            {'health': enemy_health, 'speed': enemy_speed, 'value': enemy_value} 
            for _ in range(num_enemies)
        ]
        self.wave_timer = 30 # Spawn interval

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.001 # Small penalty for time passing
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        # Wave Management
        if self.wave_state == "SPAWNING":
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.enemies_to_spawn:
                enemy_def = self.enemies_to_spawn.pop(0)
                self.enemies.append(Enemy(enemy_def['health'], enemy_def['speed'], self.path_waypoints, enemy_def['value']))
                self.wave_timer = 45 # Reset spawn interval
            elif not self.enemies_to_spawn:
                self.wave_state = "WAITING"
        
        elif self.wave_state == "WAITING":
            if not self.enemies:
                if self.wave_number >= self.TOTAL_WAVES:
                    self.win = True
                    self.game_over = True
                else:
                    self.wave_state = "PRE_WAVE"
                    self.wave_timer = 150 # 5s countdown

        elif self.wave_state == "PRE_WAVE":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        # Update Towers and fire projectiles
        for tower in self.towers:
            tower.update(self.enemies)
            if tower.can_fire():
                self.projectiles.append(Projectile(tower.pos, tower.target, tower.damage))
                tower.cooldown = tower.fire_rate
                # sfx: tower_fire.wav
                self._create_particles(tower.pos, 1, 3, (255, 255, 150), 0.5)

        # Update Projectiles
        for proj in self.projectiles[:]:
            if proj.update():
                self.projectiles.remove(proj)
                reward += 0.1 # Reward for hitting
                if proj.target.health <= 0:
                    reward += 1.0 # Reward for destroying
                    self.resources += proj.target.value
                    self._create_particles(proj.target.pos, 20, 5, self.COLOR_ENEMY, 2)
                    self.enemies.remove(proj.target)
                    # sfx: enemy_destroyed.wav
                else:
                    self._create_particles(proj.pos, 5, 2, self.COLOR_PROJECTILE, 1)
                    # sfx: enemy_hit.wav

        # Update Enemies
        for enemy in self.enemies[:]:
            if enemy.update():
                self.enemies.remove(enemy)
                self.base_health -= 10
                reward -= 10.0 # Penalty for base damage
                self._create_particles(self.base_pos, 30, 8, (255, 100, 50), 3)
                # sfx: base_damage.wav

        # Update Particles
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.win:
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        # Debounced movement
        if movement == 1: self.grid_cursor_pos[0] = max(0, self.grid_cursor_pos[0] - 1)
        elif movement == 2: self.grid_cursor_pos[0] = min(self.GRID_ROWS - 1, self.grid_cursor_pos[0] + 1)
        elif movement == 3: self.grid_cursor_pos[1] = max(0, self.grid_cursor_pos[1] - 1)
        elif movement == 4: self.grid_cursor_pos[1] = min(self.GRID_COLS - 1, self.grid_cursor_pos[1] + 1)

        # Place tower on key press
        if space_action and not self.prev_space_held:
            self._place_tower()
        
        # Cycle tower type on key press
        if shift_action and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)

        self.prev_space_held = space_action
        self.prev_shift_held = shift_action

    def _place_tower(self):
        row, col = self.grid_cursor_pos
        tower_def = self.TOWER_TYPES[self.selected_tower_idx]

        if self.placement_grid[row][col] and self.resources >= tower_def['cost']:
            self.resources -= tower_def['cost']
            self.placement_grid[row][col] = False
            
            x = self.GRID_OFFSET_X + col * self.GRID_SIZE + self.GRID_SIZE // 2
            y = self.GRID_OFFSET_Y + row * self.GRID_SIZE + self.GRID_SIZE // 2
            
            self.towers.append(Tower((x, y), tower_def))
            self._create_particles((x,y), 15, 4, tower_def['color'], 1.5)
            # sfx: place_tower.wav
    
    def _create_particles(self, pos, count, max_speed, color, size):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(10, 30)
            self.particles.append(Particle(pos, vel, size, color, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 35)
        pygame.draw.lines(self.screen, (0,0,0, 50), False, self.path_waypoints, 40) # Shadow edge
        
        # Draw Grid
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (self.GRID_OFFSET_X + c * self.GRID_SIZE, self.GRID_OFFSET_Y + r * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Draw Base
        pygame.gfxdraw.box(self.screen, pygame.Rect(self.base_pos[0]-10, self.base_pos[1]-10, 20, 20), self.COLOR_BASE)

        # Draw Towers
        for tower in self.towers:
            p1 = (tower.pos.x, tower.pos.y - 10)
            p2 = (tower.pos.x - 8, tower.pos.y + 6)
            p3 = (tower.pos.x + 8, tower.pos.y + 6)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), tower.color)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), tower.color)
            if tower.cooldown <= 0: # Glow when ready to fire
                 pygame.gfxdraw.aacircle(self.screen, int(tower.pos.x), int(tower.pos.y), 12, (*tower.color, 100))

        # Draw Enemies
        for enemy in self.enemies:
            pygame.gfxdraw.aacircle(self.screen, int(enemy.pos.x), int(enemy.pos.y), enemy.radius, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy.pos.x), int(enemy.pos.y), enemy.radius, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy.health / enemy.max_health
            bar_width = 20
            bar_height = 4
            bar_x = enemy.pos.x - bar_width // 2
            bar_y = enemy.pos.y - enemy.radius - 8
            pygame.draw.rect(self.screen, (255,0,0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0,255,0), (bar_x, bar_y, bar_width * health_pct, bar_height))

        # Draw Projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj.pos.x), int(proj.pos.y), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(proj.pos.x), int(proj.pos.y), 4, (*self.COLOR_PROJECTILE, 150))
        
        # Draw Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.size), p.color)

        # Draw Cursor
        row, col = self.grid_cursor_pos
        cursor_x = self.GRID_OFFSET_X + col * self.GRID_SIZE
        cursor_y = self.GRID_OFFSET_Y + row * self.GRID_SIZE
        tower_def = self.TOWER_TYPES[self.selected_tower_idx]
        can_afford = self.resources >= tower_def['cost']
        is_empty = self.placement_grid[row][col]
        is_valid = can_afford and is_empty
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        
        rect = pygame.Rect(cursor_x, cursor_y, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, cursor_color, rect, 2)
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, tower_def['range'], (*cursor_color, 50))


    def _render_ui(self):
        # Base Health
        health_text = self.FONT_UI.render(f"Base Health: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Resources
        res_text = self.FONT_UI.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (self.SCREEN_WIDTH - res_text.get_width() - 10, 10))
        
        # Wave Info
        wave_str = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.wave_state == "PRE_WAVE":
            wave_str += f" (Next in {self.wave_timer // 30 + 1}s)"
        wave_text = self.FONT_UI.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, self.SCREEN_HEIGHT - 30))

        # Selected Tower Info
        tower_def = self.TOWER_TYPES[self.selected_tower_idx]
        tower_info = f"Build: {tower_def['name']} (Cost: {tower_def['cost']})"
        tower_text = self.FONT_UI.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (self.SCREEN_WIDTH - tower_text.get_width() - 10, self.SCREEN_HEIGHT - 30))
        
        # Score
        score_text = self.FONT_UI.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH//2 - score_text.get_width()//2, 10))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            msg_text = self.FONT_MSG.render(msg, True, color)
            self.screen.blit(msg_text, (self.SCREEN_WIDTH//2 - msg_text.get_width()//2, self.SCREEN_HEIGHT//2 - msg_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This part requires a window and is for testing/demonstration purposes.
    # It will not work in a purely headless environment.
    try:
        import os
        # Set a video driver that can open a window
        if os.name == 'posix':
            os.environ['SDL_VIDEODRIVER'] = 'x11'
        else: # For Windows
            pass # Default should be fine
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Defense")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print(env.user_guide)

        while not done:
            # --- Action Mapping for Manual Play ---
            keys = pygame.key.get_pressed()
            move = 0 # none
            if keys[pygame.K_UP]: move = 1
            elif keys[pygame.K_DOWN]: move = 2
            elif keys[pygame.K_LEFT]: move = 3
            elif keys[pygame.K_RIGHT]: move = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Rendering to the window ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30) # Run at 30 FPS
            
        print(f"Game Over! Final Score: {info['score']}")

    except Exception as e:
        print("\nCould not create display for manual play.")
        print("This is expected in a headless environment.")
        print("The environment itself is functional.")
        print(f"Error: {e}")
    finally:
        env.close()