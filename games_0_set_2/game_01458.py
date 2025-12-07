
# Generated: 2025-08-27T17:12:35.125935
# Source Brief: brief_01458.md
# Brief Index: 1458

        
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

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press space to build a tower on a valid (yellow) spot."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of incoming enemies."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE
        
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 20
        self.INITIAL_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 150 # frames (5 seconds at 30fps)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 70)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_TOWER_ZONE = (255, 255, 25, 50)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_TOWER = (0, 130, 200)
        self.COLOR_PROJECTILE = (70, 240, 240)
        self.COLOR_CURSOR_VALID = (0, 255, 0, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.COLOR_UI_TEXT = (245, 245, 245)
        self.COLOR_HEALTH_GREEN = (50, 205, 50)
        self.COLOR_HEALTH_RED = (220, 20, 60)
        
        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 52)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.wave = 0
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.space_was_pressed = False
        self.rng = None

        # --- World Definition ---
        self._define_world()

        # Initialize state variables
        self.reset()
        
        # --- Critical Self-Check ---
        self.validate_implementation()
    
    def _define_world(self):
        """Defines the static elements of the game world like path and tower zones."""
        self.path_nodes = [
            (2, -1), (2, 2), (13, 2), (13, 7), (6, 7), (6, 11)
        ]
        self.base_pos = self.path_nodes[-2]

        self.tower_zones = [
            (4, 1), (8, 1), (11, 1),
            (14, 4), (12, 4), (10, 4),
            (5, 6), (8, 6), (5, 8),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.wave = 0 # Will be incremented to 1 by wave manager
        self.wave_timer = self.WAVE_PREP_TIME - 1 # Start first wave almost immediately

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.space_was_pressed = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty for each step to encourage efficiency

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            reward += self._update_wave_manager()
            self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
            elif self.base_health <= 0:
                reward += -10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right

        # Wrap cursor around screen
        self.cursor_pos[0] %= self.GRID_WIDTH
        self.cursor_pos[1] %= self.GRID_HEIGHT

        # Place tower on space press (rising edge)
        if space_held and not self.space_was_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            is_valid_zone = cursor_tuple in self.tower_zones
            is_occupied = any(t['pos'] == self.cursor_pos for t in self.towers)
            
            if is_valid_zone and not is_occupied:
                self.towers.append({
                    'pos': list(self.cursor_pos),
                    'range': 120, # pixels
                    'fire_rate': 45, # frames
                    'cooldown': 0,
                    'flash': 0
                })
                # sfx_place_tower

        self.space_was_pressed = space_held

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            tower['flash'] = max(0, tower['flash'] - 1)
            
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    tower_px = self._grid_to_pixel_center(tower['pos'])
                    self.projectiles.append({
                        'pos': list(tower_px),
                        'target': target,
                        'speed': 6,
                        'damage': 5
                    })
                    tower['cooldown'] = tower['fire_rate']
                    tower['flash'] = 5
                    # sfx_tower_shoot
        return 0

    def _find_target(self, tower):
        tower_px = self._grid_to_pixel_center(tower['pos'])
        
        # Find the enemy that has progressed furthest down the path
        best_target = None
        max_progress = -1

        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - tower_px[0], enemy['pos'][1] - tower_px[1])
            if dist <= tower['range']:
                enemy_progress = enemy['path_index'] + enemy['progress']
                if enemy_progress > max_progress:
                    max_progress = enemy_progress
                    best_target = enemy
        return best_target

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_pos = proj['target']['pos']
            direction = math.atan2(target_pos[1] - proj['pos'][1], target_pos[0] - proj['pos'][0])
            proj['pos'][0] += math.cos(direction) * proj['speed']
            proj['pos'][1] += math.sin(direction) * proj['speed']
            
            if math.hypot(proj['pos'][0] - target_pos[0], proj['pos'][1] - target_pos[1]) < 10:
                proj['target']['health'] -= proj['damage']
                self._create_particles(target_pos, self.COLOR_PROJECTILE, 10, 2)
                self.projectiles.remove(proj)
                reward += 0.1 # Reward for hitting an enemy
                # sfx_enemy_hit
        return reward
    
    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 30, 4)
                self.enemies.remove(enemy)
                self.score += 10
                reward += 1 # Reward for defeating an enemy
                # sfx_enemy_destroyed
                continue

            # Move enemy along path
            enemy['progress'] += enemy['speed']
            if enemy['progress'] >= 1.0:
                enemy['progress'] = 0.0
                enemy['path_index'] += 1

                if enemy['path_index'] >= len(self.path_nodes) - 1:
                    self.base_health = max(0, self.base_health - 10)
                    self._create_particles(self._grid_to_pixel_center(self.base_pos), self.COLOR_BASE, 50, 5)
                    self.enemies.remove(enemy)
                    # sfx_base_damage
                    continue
            
            # Interpolate position for smooth movement
            p1 = self._grid_to_pixel_center(self.path_nodes[enemy['path_index']])
            p2 = self._grid_to_pixel_center(self.path_nodes[enemy['path_index'] + 1])
            enemy['pos'][0] = p1[0] + (p2[0] - p1[0]) * enemy['progress']
            enemy['pos'][1] = p1[1] + (p2[1] - p1[1]) * enemy['progress']
        
        return reward

    def _update_wave_manager(self):
        reward = 0
        if not self.enemies and self.wave < self.MAX_WAVES:
            self.wave_timer += 1
            if self.wave_timer >= self.WAVE_PREP_TIME:
                if self.wave > 0:
                    reward += 50 # Reward for surviving a wave
                    self.score += 100
                self.wave += 1
                self._spawn_wave()
                self.wave_timer = 0
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.wave
        enemy_health = 10 + self.wave * 5
        enemy_speed = 0.015 + self.wave * 0.0005
        
        start_pos = self._grid_to_pixel_center(self.path_nodes[0])
        for i in range(num_enemies):
            self.enemies.append({
                'path_index': 0,
                'progress': -0.2 * i, # Stagger spawn
                'pos': list(start_pos),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'id': self.rng.integers(100000)
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.wave > self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw path
        for i in range(len(self.path_nodes) - 1):
            p1 = self._grid_to_pixel_center(self.path_nodes[i])
            p2 = self._grid_to_pixel_center(self.path_nodes[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.GRID_SIZE)
        
        # Draw tower placement zones
        for zone_pos in self.tower_zones:
            px_pos = self._grid_to_pixel_center(zone_pos)
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_TOWER_ZONE)
            self.screen.blit(s, (px_pos[0] - self.GRID_SIZE // 2, px_pos[1] - self.GRID_SIZE // 2))

        # Draw base
        base_px = self._grid_to_pixel_center(self.base_pos)
        pygame.gfxdraw.filled_circle(self.screen, base_px[0], base_px[1], self.GRID_SIZE // 2, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_px[0], base_px[1], self.GRID_SIZE // 2, self.COLOR_BASE)

    def _render_entities(self):
        # Draw towers
        for tower in self.towers:
            px_pos = self._grid_to_pixel_center(tower['pos'])
            if tower['flash'] > 0:
                flash_alpha = (tower['flash'] / 5) * 200
                pygame.gfxdraw.filled_circle(self.screen, px_pos[0], px_pos[1], 18, (*self.COLOR_PROJECTILE, int(flash_alpha)))
            pygame.gfxdraw.filled_circle(self.screen, px_pos[0], px_pos[1], 12, self.COLOR_TOWER)
            pygame.gfxdraw.aacircle(self.screen, px_pos[0], px_pos[1], 12, self.COLOR_TOWER)

        # Draw enemies
        for enemy in self.enemies:
            px, py = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, px, py, 8, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            bar_height = 3
            bar_x = px - bar_width // 2
            bar_y = py - 15
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

        # Draw projectiles
        for proj in self.projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, 4, self.COLOR_PROJECTILE)

    def _render_effects(self):
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color_with_alpha = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))

        # Draw cursor
        cursor_px = self._grid_to_pixel_center(self.cursor_pos)
        cursor_tuple = tuple(self.cursor_pos)
        is_valid = cursor_tuple in self.tower_zones and not any(t['pos'] == self.cursor_pos for t in self.towers)
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, (0, 0, self.GRID_SIZE, self.GRID_SIZE), 0, 4)
        pygame.draw.rect(s, (255,255,255, cursor_color[3]+50), (0, 0, self.GRID_SIZE, self.GRID_SIZE), 2, 4)
        self.screen.blit(s, (cursor_px[0] - self.GRID_SIZE // 2, cursor_px[1] - self.GRID_SIZE // 2))

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))

        # Base Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_bar_x = (self.SCREEN_WIDTH - health_bar_width) // 2
        health_bar_y = self.SCREEN_HEIGHT - health_bar_height - 5
        health_ratio = self.base_health / self.INITIAL_BASE_HEALTH
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (health_bar_x, health_bar_y, int(health_bar_width * health_ratio), health_bar_height), border_radius=4)
        health_text = self.font_main.render("BASE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (health_bar_x + (health_bar_width - health_text.get_width()) // 2, health_bar_y))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "enemies_remaining": len(self.enemies),
            "towers_built": len(self.towers)
        }
    
    def _grid_to_pixel_center(self, grid_pos):
        x = grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2
        y = grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        return [x, y]

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': life,
                'max_life': life,
                'size': random.randint(1,4)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # --- Human Input ---
        movement_action = 0 # No movement
        space_action = 0 # Not pressed
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first key found
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        # This is for quitting the window, not part of the env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        action = [movement_action, space_action, 0] # Shift is not used
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Final Info: {info}")
            # Optional: auto-reset after a delay
            # pygame.time.wait(2000)
            # obs, info = env.reset()
            # total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)
        
    env.close()