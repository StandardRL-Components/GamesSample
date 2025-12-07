
# Generated: 2025-08-27T18:13:32.604303
# Source Brief: brief_01766.md
# Brief Index: 1766

        
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
        "Controls: Arrow keys to move cursor. Space to place a tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 30 * 60  # 60 seconds at 30fps
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (40, 40, 55)
    COLOR_GRID = (50, 50, 70)
    COLOR_BASE = (0, 180, 120)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_ACCENT = (100, 100, 255)
    COLOR_CURSOR = (255, 255, 255)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game path and grid setup
        self.path_waypoints = [
            (-20, 200), (100, 200), (100, 100), (300, 100),
            (300, 300), (540, 300), (540, 150), (self.SCREEN_WIDTH + 20, 150)
        ]
        self.base_pos = (self.SCREEN_WIDTH, 150)
        self.base_rect = pygame.Rect(self.SCREEN_WIDTH - 20, 130, 20, 40)

        self.tower_spots = []
        for y in range(50, self.SCREEN_HEIGHT, 80):
            for x in range(50, self.SCREEN_WIDTH, 80):
                 if not self._is_point_on_path((x, y), 30):
                    self.tower_spots.append((x, y))

        # Tower definitions
        self.tower_types = [
            {"name": "Gun Turret", "cost": 100, "range": 70, "damage": 10, "fire_rate": 0.8, "color": (80, 120, 255), "projectile_speed": 8},
            {"name": "Sniper", "cost": 250, "range": 150, "damage": 50, "fire_rate": 2.0, "color": (150, 100, 255), "projectile_speed": 15},
        ]
        
        # State variables are initialized in reset()
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 100
        self.money = 250
        
        # Wave management
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_spawn_timer = 0
        self.enemies_to_spawn = 0

        # Entities
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Player control state
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.step_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0

        self._handle_input(action)
        self._update_game_logic()
        
        self.steps += 1
        reward = self.step_reward
        terminated = self._check_termination()
        
        if terminated:
            if self.victory:
                reward += 100
            else: # Loss by health or time
                reward -= 100

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
        if movement != 0:
            current_pos = self.tower_spots[self.cursor_index]
            best_next_index = self.cursor_index
            min_dist_sq = float('inf')

            for i, pos in enumerate(self.tower_spots):
                if i == self.cursor_index: continue
                dx, dy = pos[0] - current_pos[0], pos[1] - current_pos[1]
                
                # Check if in correct direction
                is_correct_dir = False
                if movement == 1 and dy < 0 and abs(dy) > abs(dx): is_correct_dir = True # Up
                elif movement == 2 and dy > 0 and abs(dy) > abs(dx): is_correct_dir = True # Down
                elif movement == 3 and dx < 0 and abs(dx) > abs(dy): is_correct_dir = True # Left
                elif movement == 4 and dx > 0 and abs(dx) > abs(dy): is_correct_dir = True # Right
                
                if is_correct_dir:
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_next_index = i
            self.cursor_index = best_next_index

        # --- Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()

        # --- Cycle Tower Type (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        spot = self.tower_spots[self.cursor_index]
        tower_def = self.tower_types[self.selected_tower_type]
        
        # Check if spot is occupied
        is_occupied = any(t['pos'] == spot for t in self.towers)
        
        if not is_occupied and self.money >= tower_def['cost']:
            self.money -= tower_def['cost']
            self.towers.append({
                "pos": spot,
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            })
            # Sound: # Tower placement sound

    def _update_game_logic(self):
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()

    def _update_waves(self):
        if not self.wave_in_progress and not self.enemies:
            if self.wave_number >= self.MAX_WAVES:
                self.victory = True
                return

            # Start next wave
            if self.wave_number > 0:
                self.step_reward += 1.0 # Wave complete bonus
            self.wave_number += 1
            self.wave_in_progress = True
            self.enemies_to_spawn = 2 + self.wave_number
            self.wave_spawn_timer = 1.0 * self.FPS # 1 second delay

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            tower_def = self.tower_types[tower['type']]
            
            if tower['cooldown'] == 0:
                # Find target
                target = None
                min_dist_sq = tower_def['range'] ** 2
                for enemy in self.enemies:
                    dist_sq = (tower['pos'][0] - enemy['pos'][0])**2 + (tower['pos'][1] - enemy['pos'][1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target = enemy
                
                if target:
                    tower['target'] = target
                    self.projectiles.append({
                        "pos": list(tower['pos']),
                        "tower_type": tower['type'],
                        "target": target,
                    })
                    tower['cooldown'] = tower_def['fire_rate'] * self.FPS
                    # Sound: # Tower fire sound

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj['target']
            tower_def = self.tower_types[proj['tower_type']]

            # If target is dead, remove projectile
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            # Move towards target
            direction = np.array(target['pos']) - np.array(proj['pos'])
            dist = np.linalg.norm(direction)
            if dist < tower_def['projectile_speed']:
                # Hit
                self._handle_hit(proj)
                self.projectiles.remove(proj)
            else:
                # Move
                direction = direction / dist
                proj['pos'][0] += direction[0] * tower_def['projectile_speed']
                proj['pos'][1] += direction[1] * tower_def['projectile_speed']

    def _handle_hit(self, proj):
        enemy = proj['target']
        tower_def = self.tower_types[proj['tower_type']]
        
        enemy['health'] -= tower_def['damage']
        # Sound: # Projectile hit sound
        
        # Hit particle effect
        for _ in range(5):
            self.particles.append(self._create_particle(enemy['pos'], (255, 255, 200), 1, 3, 5))

        if enemy['health'] <= 0:
            # Sound: # Enemy death sound
            # Death particle effect
            for _ in range(20):
                self.particles.append(self._create_particle(enemy['pos'], (255, 100, 100), 2, 5, 15))
            
            self.enemies.remove(enemy)
            self.money += 15 + self.wave_number * 2
            self.score += 10
            self.step_reward += 0.1

    def _update_enemies(self):
        if self.wave_in_progress and self.wave_spawn_timer > 0:
            self.wave_spawn_timer -= 1
        elif self.wave_in_progress and self.enemies_to_spawn > 0:
            # Spawn enemy
            speed = 1.0 + self.wave_number * 0.05
            health = 50 + (self.wave_number - 1) * 20
            self.enemies.append({
                'pos': list(self.path_waypoints[0]),
                'path_index': 1,
                'speed': speed,
                'health': health,
                'max_health': health,
            })
            self.enemies_to_spawn -= 1
            self.wave_spawn_timer = 0.5 * self.FPS # spawn interval
            if self.enemies_to_spawn == 0:
                self.wave_in_progress = False

        for enemy in self.enemies[:]:
            target_pos = self.path_waypoints[enemy['path_index']]
            direction = np.array(target_pos) - np.array(enemy['pos'])
            dist = np.linalg.norm(direction)

            if dist < enemy['speed']:
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path_waypoints):
                    # Reached base
                    self.base_health = max(0, self.base_health - 10)
                    self.step_reward -= 1.0 # -0.1 per health point * 10
                    self.enemies.remove(enemy)
                    # Sound: # Base damage alert
                    # Create base hit particle effect (screen flash)
                    self.particles.append({"type": "flash", "color": (255, 0, 0, 100), "duration": 5, "max_duration": 5})
                    continue
            else:
                # Move
                move_vec = direction / dist * enemy['speed']
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]
    
    def _create_particle(self, pos, color, speed, size, duration):
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
               math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
        return {"pos": list(pos), "vel": vel, "color": color, "size": size, "duration": duration, "max_duration": duration, "type": "spark"}

    def _update_particles(self):
        for p in self.particles[:]:
            p['duration'] -= 1
            if p['duration'] <= 0:
                self.particles.remove(p)
                continue
            if p['type'] == 'spark':
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['size'] *= 0.95

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0:
            self.game_over = True
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
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 40)
        # Draw grid spots
        for spot in self.tower_spots:
            pygame.gfxdraw.filled_circle(self.screen, spot[0], spot[1], 3, self.COLOR_GRID)
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)

        # Draw towers
        for tower in self.towers:
            tower_def = self.tower_types[tower['type']]
            x, y = tower['pos']
            color = tower_def['color']
            if tower['type'] == 0: # Gun Turret
                pygame.draw.circle(self.screen, color, (x,y), 10)
                pygame.draw.circle(self.screen, self.COLOR_BG, (x,y), 6)
            elif tower['type'] == 1: # Sniper
                points = [(x, y - 10), (x - 8, y + 8), (x + 8, y + 8)]
                pygame.draw.polygon(self.screen, color, points)

        # Draw projectiles
        for proj in self.projectiles:
            tower_def = self.tower_types[proj['tower_type']]
            color = (255, 255, 100) if tower_def['name'] == "Gun Turret" else (255, 180, 255)
            pygame.draw.circle(self.screen, color, (int(proj['pos'][0]), int(proj['pos'][1])), 3)

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            bar_x = x - bar_width // 2
            bar_y = y - 15
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, 3))
            pygame.draw.rect(self.screen, (0,255,0) if health_ratio > 0.5 else (255,255,0) if health_ratio > 0.2 else (255,0,0), (bar_x, bar_y, int(bar_width * health_ratio), 3))

        # Draw cursor and tower range
        cursor_pos = self.tower_spots[self.cursor_index]
        tower_def = self.tower_types[self.selected_tower_type]
        # Range circle
        range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        can_afford = self.money >= tower_def['cost']
        range_color = (255, 255, 255, 30) if can_afford else (255, 0, 0, 30)
        pygame.gfxdraw.filled_circle(range_surf, cursor_pos[0], cursor_pos[1], tower_def['range'], range_color)
        self.screen.blit(range_surf, (0,0))
        # Cursor rectangle
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_pos[0]-15, cursor_pos[1]-15, 30, 30), 2)
        
        # Draw particles
        for p in self.particles:
            if p['type'] == 'spark':
                alpha = int(255 * (p['duration'] / p['max_duration']))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0,0, p['size']*2, p['size']*2))
                self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))
            elif p['type'] == 'flash':
                alpha = int(p['color'][3] * (p['duration'] / p['max_duration']))
                flash_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                flash_surf.fill((*p['color'][:3], alpha))
                self.screen.blit(flash_surf, (0,0))

    def _render_ui(self):
        # Wave number
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Money
        money_text = self.font_small.render(f"$ {self.money}", True, (255, 223, 0))
        self.screen.blit(money_text, (self.SCREEN_WIDTH // 2 - money_text.get_width() // 2, 10))
        
        # Base Health
        health_text = self.font_small.render("Base Health", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))
        health_bar_rect = pygame.Rect(self.SCREEN_WIDTH - 110, 35, 100, 15)
        pygame.draw.rect(self.screen, (50, 0, 0), health_bar_rect)
        health_ratio = self.base_health / 100
        pygame.draw.rect(self.screen, self.COLOR_BASE, (health_bar_rect.x, health_bar_rect.y, health_bar_rect.width * health_ratio, health_bar_rect.height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_bar_rect, 1)

        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - 50))
        
        # Selected Tower Info
        tower_def = self.tower_types[self.selected_tower_type]
        can_afford = self.money >= tower_def['cost']
        tower_name_color = self.COLOR_TEXT if can_afford else (255, 100, 100)
        tower_name = self.font_small.render(f"Build: {tower_def['name']}", True, tower_name_color)
        tower_cost = self.font_small.render(f"Cost: ${tower_def['cost']}", True, tower_name_color)
        self.screen.blit(tower_name, (10, self.SCREEN_HEIGHT - 50))
        self.screen.blit(tower_cost, (10, self.SCREEN_HEIGHT - 30))

        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - 24))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "money": self.money,
            "base_health": self.base_health,
        }
        
    def _is_point_on_path(self, point, threshold):
        px, py = point
        for i in range(len(self.path_waypoints) - 1):
            p1 = np.array(self.path_waypoints[i])
            p2 = np.array(self.path_waypoints[i+1])
            line_vec = p2 - p1
            point_vec = np.array(point) - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue
            
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            closest_point = p1 + t * line_vec
            dist_sq = (closest_point[0] - px)**2 + (closest_point[1] - py)**2
            if dist_sq < threshold**2:
                return True
        return False
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
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
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)
        
    env.close()