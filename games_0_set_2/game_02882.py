
# Generated: 2025-08-27T21:43:07.555261
# Source Brief: brief_02882.md
# Brief Index: 2882

        
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
        "Controls: Use arrow keys to move the placement cursor. Press space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    TILE_SIZE = 20

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_GRID = (30, 40, 50)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_GLOW = (0, 200, 255)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_HEALTH = (50, 200, 50)
    COLOR_TOWER = (0, 200, 100)
    COLOR_TOWER_RANGE = (100, 120, 100, 100) # RGBA
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_CURSOR_VALID = (0, 255, 0, 100) # RGBA
    COLOR_CURSOR_INVALID = (255, 0, 0, 100) # RGBA
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (50, 60, 70, 180) # RGBA

    # Game parameters
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 15
    MAX_WAVES = 5
    INTERMISSION_TIME = 150 # 5 seconds at 30fps
    TOWER_COST = 5
    TOWER_RANGE_SQ = 80**2 # pixels squared for efficiency
    TOWER_FIRE_RATE = 15 # frames (2 times per second)
    ENEMY_START_HEALTH = 3
    ENEMY_REWARD = 1
    WAVE_SURVIVAL_REWARD = 10
    HIT_REWARD = 0.1
    VICTORY_REWARD = 100
    DEFEAT_REWARD = -100
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.game_phase = "intermission" # "intermission" or "wave_active"
        self.intermission_timer = self.INTERMISSION_TIME
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self._generate_path()
        
        self.cursor_pos = [self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2]
        self.last_space_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        step_reward = 0

        self._update_cursor(movement)
        
        if self.game_phase == "intermission":
            self._handle_placement(space_pressed)
            self.intermission_timer -= 1
            if self.intermission_timer <= 0:
                self.current_wave += 1
                self._spawn_wave()
                self.game_phase = "wave_active"
        elif self.game_phase == "wave_active":
            self._handle_placement(space_pressed)
            step_reward += self._update_towers()
            step_reward += self._update_projectiles()
            self._update_enemies()
            
            if not self.enemies:
                self.game_phase = "intermission"
                self.intermission_timer = self.INTERMISSION_TIME
                step_reward += self.WAVE_SURVIVAL_REWARD
                # sfx: wave_complete.wav

        self._update_particles()
        self.steps += 1
        self.score += step_reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0:
                step_reward += self.DEFEAT_REWARD
            elif self.current_wave >= self.MAX_WAVES:
                step_reward += self.VICTORY_REWARD
            self.score += step_reward # Add terminal reward to final score

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Update Logic ---
    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _handle_placement(self, space_pressed):
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            is_on_path = cursor_tuple in self.path_tiles
            is_occupied = any(t['pos_grid'] == cursor_tuple for t in self.towers)
            
            if self.resources >= self.TOWER_COST and not is_on_path and not is_occupied:
                self.resources -= self.TOWER_COST
                self.towers.append({
                    "pos_grid": cursor_tuple,
                    "pos_pixel": (cursor_tuple[0] * self.TILE_SIZE + self.TILE_SIZE // 2, 
                                  cursor_tuple[1] * self.TILE_SIZE + self.TILE_SIZE // 2),
                    "cooldown": 0, "pulse": 0
                })
                # sfx: place_tower.wav

    def _update_towers(self):
        for tower in self.towers:
            tower["pulse"] = (tower["pulse"] + 0.1) % (2 * math.pi)
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            target, min_dist_sq = None, self.TOWER_RANGE_SQ
            for enemy in self.enemies:
                if enemy['spawn_delay'] > 0: continue
                dist_sq = (enemy['pos'][0] - tower['pos_pixel'][0])**2 + (enemy['pos'][1] - tower['pos_pixel'][1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = enemy
            
            if target:
                self.projectiles.append({
                    "pos": list(tower['pos_pixel']),
                    "target_pos": list(target['pos']),
                    "speed": 5,
                })
                tower['cooldown'] = self.TOWER_FIRE_RATE
                # sfx: tower_shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            direction = np.array(p['target_pos']) - np.array(p['pos'])
            distance = np.linalg.norm(direction)
            
            if distance < p['speed']:
                if p in self.projectiles: self.projectiles.remove(p)
                continue
            
            move_vec = (direction / distance) * p['speed']
            p['pos'][0] += move_vec[0]
            p['pos'][1] += move_vec[1]
            
            for enemy in self.enemies[:]:
                if enemy['spawn_delay'] > 0: continue
                dist_sq = (p['pos'][0] - enemy['pos'][0])**2 + (p['pos'][1] - enemy['pos'][1])**2
                if dist_sq < (self.TILE_SIZE * 0.4)**2:
                    enemy['health'] -= 1
                    reward += self.HIT_REWARD
                    if p in self.projectiles: self.projectiles.remove(p)
                    # sfx: enemy_hit.wav
                    
                    if enemy['health'] <= 0:
                        reward += self.ENEMY_REWARD
                        self.resources += self.ENEMY_REWARD
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 20)
                        if enemy in self.enemies: self.enemies.remove(enemy)
                        # sfx: enemy_destroy.wav
                    break
        return reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['spawn_delay'] > 0:
                enemy['spawn_delay'] -= 1
                continue

            if enemy['waypoint_index'] >= len(self.path_waypoints):
                self.base_health = max(0, self.base_health - 1)
                self._create_particles(self.base_pos_pixel, self.COLOR_ENEMY, 10)
                if enemy in self.enemies: self.enemies.remove(enemy)
                # sfx: base_damage.wav
                continue
            
            target_pos = self.path_waypoints[enemy['waypoint_index']]
            direction = np.array(target_pos) - np.array(enemy['pos'])
            distance = np.linalg.norm(direction)
            
            if distance < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['waypoint_index'] += 1
            else:
                move_vec = (direction / distance) * enemy['speed']
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                if p in self.particles: self.particles.remove(p)

    # --- State & Game Flow ---
    def _generate_path(self):
        start_y = self.np_random.integers(3, self.GRID_HEIGHT - 3)
        self.base_pos_grid = (self.GRID_WIDTH - 2, self.np_random.integers(3, self.GRID_HEIGHT - 3))
        
        path_grid = []
        current_pos = [1, start_y]
        path_grid.append(list(current_pos))

        while current_pos[0] < self.base_pos_grid[0]:
            move_x = self.np_random.choice([0, 1], p=[0.3, 0.7]) if current_pos[0] < self.base_pos_grid[0] - 1 else 1
            if move_x == 1:
                current_pos[0] += 1
            else:
                move_y = self.np_random.choice([-1, 1])
                current_pos[1] = np.clip(current_pos[1] + move_y, 1, self.GRID_HEIGHT - 2)
            if path_grid[-1] != current_pos:
                path_grid.append(list(current_pos))
        
        self.path_waypoints = [(x * self.TILE_SIZE + self.TILE_SIZE // 2, y * self.TILE_SIZE + self.TILE_SIZE // 2) for x, y in path_grid]
        self.path_tiles = set(tuple(p) for p in path_grid)
        self.base_pos_pixel = (self.base_pos_grid[0] * self.TILE_SIZE + self.TILE_SIZE // 2, self.base_pos_grid[1] * self.TILE_SIZE + self.TILE_SIZE // 2)

    def _spawn_wave(self):
        num_enemies = 5 + self.current_wave
        enemy_speed = 1.0 + (self.current_wave - 1) * 0.1
        for i in range(num_enemies):
            self.enemies.append({
                "pos": list(self.path_waypoints[0]), "waypoint_index": 1,
                "health": self.ENEMY_START_HEALTH + self.current_wave,
                "max_health": self.ENEMY_START_HEALTH + self.current_wave,
                "speed": enemy_speed, "spawn_delay": i * 30,
            })
        # sfx: wave_start.wav

    def _check_termination(self):
        if self.base_health <= 0: return True
        if self.current_wave >= self.MAX_WAVES and not self.enemies: return True
        if self.steps >= self.MAX_STEPS: return True
        return False

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                "radius": self.np_random.uniform(2, 5),
                "color": color, "life": 20
            })
            
    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_path()
        self._render_base()
        for tower in self.towers: self._render_tower(tower)
        for p in self.projectiles: self._render_projectile(p)
        for enemy in self.enemies:
            if enemy['spawn_delay'] <= 0: self._render_enemy(enemy)
        self._render_particles()
        self._render_cursor()

    def _render_path(self):
        if len(self.path_waypoints) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 1)
            for point in self.path_waypoints:
                pygame.gfxdraw.filled_circle(self.screen, int(point[0]), int(point[1]), self.TILE_SIZE // 4, self.COLOR_PATH)

    def _render_base(self):
        pos = self.base_pos_pixel
        size = self.TILE_SIZE
        glow = 5 + 3 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.box(self.screen, (pos[0] - size//2, pos[1] - size//2, size, size), self.COLOR_BASE)
        pygame.gfxdraw.rectangle(self.screen, (pos[0] - size//2 - int(glow)//2, pos[1] - size//2- int(glow)//2, size+int(glow), size+int(glow)), (*self.COLOR_BASE_GLOW, 50))
    
    def _render_tower(self, tower):
        pos = tower['pos_pixel']
        size = self.TILE_SIZE * 0.8
        points = [
            (pos[0], pos[1] - size // 2),
            (pos[0] - size // 2, pos[1] + size // 2),
            (pos[0] + size // 2, pos[1] + size // 2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TOWER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TOWER)
        
        # Range indicator when cursor is over
        cursor_dist_sq = (self.cursor_pos[0]*self.TILE_SIZE - pos[0])**2 + (self.cursor_pos[1]*self.TILE_SIZE - pos[1])**2
        if cursor_dist_sq < (self.TILE_SIZE)**2:
            range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(range_surf, pos[0], pos[1], int(math.sqrt(self.TOWER_RANGE_SQ)), self.COLOR_TOWER_RANGE)
            self.screen.blit(range_surf, (0,0))
    
    def _render_enemy(self, enemy):
        pos_x, pos_y = int(enemy['pos'][0]), int(enemy['pos'][1])
        radius = int(self.TILE_SIZE * 0.4)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, self.COLOR_ENEMY)
        # Health bar
        health_ratio = enemy['health'] / enemy['max_health']
        bar_width = self.TILE_SIZE
        bar_height = 4
        bar_x = pos_x - bar_width // 2
        bar_y = pos_y - radius - 8
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

    def _render_projectile(self, p):
        pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, self.COLOR_PROJECTILE)
        pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), max(0, int(p['radius'])), p['color'])

    def _render_cursor(self):
        cursor_tuple = tuple(self.cursor_pos)
        is_on_path = cursor_tuple in self.path_tiles
        is_occupied = any(t['pos_grid'] == cursor_tuple for t in self.towers)
        can_afford = self.resources >= self.TOWER_COST
        
        is_valid = not is_on_path and not is_occupied and can_afford
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        cursor_rect = (self.cursor_pos[0] * self.TILE_SIZE, self.cursor_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        
        cursor_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(color)
        self.screen.blit(cursor_surface, cursor_rect[:2])

    def _render_ui(self):
        panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surface, panel_rect.topleft)

        texts = [
            f"Base HP: {self.base_health}/{self.BASE_START_HEALTH}",
            f"Resources: {self.resources}",
            f"Wave: {self.current_wave}/{self.MAX_WAVES}",
            f"Score: {int(self.score)}"
        ]
        positions = [(10, self.SCREEN_HEIGHT - 30), (200, self.SCREEN_HEIGHT - 30), (380, self.SCREEN_HEIGHT - 30), (520, self.SCREEN_HEIGHT - 30)]

        for text, pos in zip(texts, positions):
            rendered_text = self.font_small.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(rendered_text, pos)
            
        if self.game_phase == "intermission" and not self.game_over:
            text_str = f"Wave {self.current_wave + 1} starting..." if self.current_wave < self.MAX_WAVES else "VICTORY!"
            color = self.COLOR_UI_TEXT if self.current_wave < self.MAX_WAVES else self.COLOR_BASE_GLOW
            intermission_text = self.font_large.render(text_str, True, color)
            text_rect = intermission_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(intermission_text, text_rect)
            
        if self.game_over:
            text_str = "GAME OVER" if self.base_health <= 0 else "VICTORY!"
            color = self.COLOR_ENEMY if self.base_health <= 0 else self.COLOR_BASE_GLOW
            end_text = self.font_large.render(text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "resources": self.resources, "base_health": self.base_health}

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # No-op
    
    print("\n" + env.game_description)
    print(env.user_guide)

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()
    print(f"Game Over! Final Info: {info}")