
# Generated: 2025-08-28T04:05:49.481203
# Source Brief: brief_05144.md
# Brief Index: 5144

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Collect all yellow gems to win. Avoid the enemies!"
    )

    game_description = (
        "Collect sparkling gems while dodging cunning enemies in a fast-paced, top-down arcade environment."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.ENEMY_SIZE = 22
        self.GEM_SIZE = 12
        self.MAX_STEPS = 1000
        self.NUM_GEMS = 20
        self.NUM_ENEMY_RED = 3
        self.NUM_ENEMY_PURPLE = 3
        self.NUM_ENEMY_ORANGE = 2
        self.MAX_HEALTH = 10
        self.INVULNERABILITY_DURATION = 60

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_ENEMY_RED = (255, 50, 50)
        self.COLOR_ENEMY_PURPLE = (200, 50, 255)
        self.COLOR_ENEMY_ORANGE = (255, 150, 0)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (50, 50, 50)
        self.COLOR_HEALTH_BAR_FG = (0, 220, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_health = 0
        self.invulnerability_timer = 0
        self.gems = []
        self.enemies = []
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_health = self.MAX_HEALTH
        self.invulnerability_timer = 0
        
        self.particles = []
        
        # Spawn Gems
        self.gems = []
        while len(self.gems) < self.NUM_GEMS:
            pos = np.array([
                self.np_random.uniform(self.GEM_SIZE, self.WIDTH - self.GEM_SIZE),
                self.np_random.uniform(self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE)
            ])
            if np.linalg.norm(pos - self.player_pos) > self.PLAYER_SIZE * 3:
                self.gems.append(pos)
        
        # Spawn Enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMY_RED):
            self._spawn_enemy("red")
        for _ in range(self.NUM_ENEMY_PURPLE):
            self._spawn_enemy("purple")
        for _ in range(self.NUM_ENEMY_ORANGE):
            self._spawn_enemy("orange")

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1 # Per-step penalty for efficiency

        # --- Player Movement ---
        dist_before = self._get_dist_to_nearest_gem()
        
        player_velocity = np.array([0.0, 0.0])
        if movement == 1: player_velocity[1] = -1
        elif movement == 2: player_velocity[1] = 1
        elif movement == 3: player_velocity[0] = -1
        elif movement == 4: player_velocity[0] = 1
        
        if np.linalg.norm(player_velocity) > 0:
            player_velocity = player_velocity / np.linalg.norm(player_velocity)
        
        self.player_pos += player_velocity * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
        
        dist_after = self._get_dist_to_nearest_gem()

        if dist_after < dist_before:
            reward += 0.2

        # --- Update Game State ---
        self.steps += 1
        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1
            
        self._update_enemies()
        self._update_particles()
        
        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Gems
        gems_collected_indices = []
        for i, gem_pos in enumerate(self.gems):
            gem_rect = pygame.Rect(gem_pos[0] - self.GEM_SIZE / 2, gem_pos[1] - self.GEM_SIZE / 2, self.GEM_SIZE, self.GEM_SIZE)
            if player_rect.colliderect(gem_rect):
                gems_collected_indices.append(i)
                self.score += 10
                reward += 10
                # # GEM_COLLECT_SOUND
                
                # Risk/reward bonus
                is_enemy_close = any(np.linalg.norm(self.player_pos - e['pos']) < self.ENEMY_SIZE * 3 for e in self.enemies)
                if is_enemy_close:
                    reward += 2
                    self.score += 2

                self._create_particles(gem_pos, self.COLOR_GEM, 15)

        for i in sorted(gems_collected_indices, reverse=True):
            del self.gems[i]

        # Enemies
        if self.invulnerability_timer == 0:
            for enemy in self.enemies:
                enemy_rect = pygame.Rect(enemy['pos'][0] - self.ENEMY_SIZE / 2, enemy['pos'][1] - self.ENEMY_SIZE / 2, self.ENEMY_SIZE, self.ENEMY_SIZE)
                if player_rect.colliderect(enemy_rect):
                    self.player_health -= enemy['damage']
                    reward -= 5
                    self.invulnerability_timer = self.INVULNERABILITY_DURATION
                    # # PLAYER_HIT_SOUND
                    break
        
        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
            # # GAME_OVER_SOUND
        elif not self.gems:
            terminated = True
            self.game_over = True
            reward += 100
            self.score += 100
            # # VICTORY_SOUND
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "health": self.player_health,
            "gems_remaining": len(self.gems),
        }

    # --- Helper and Rendering Methods ---

    def _spawn_enemy(self, type):
        pos = np.array([
            self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE),
            self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
        ])
        while np.linalg.norm(pos - self.player_pos) < self.PLAYER_SIZE * 5:
            pos = np.array([
                self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE),
                self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
            ])

        if type == "red":
            self.enemies.append({
                'type': 'red', 'pos': pos, 'color': self.COLOR_ENEMY_RED,
                'speed': self.PLAYER_SPEED * 0.4, 'damage': 1,
                'dir': 1 if self.np_random.random() > 0.5 else -1
            })
        elif type == "purple":
            self.enemies.append({
                'type': 'purple', 'pos': pos, 'color': self.COLOR_ENEMY_PURPLE,
                'speed': self.PLAYER_SPEED * 0.4, 'damage': 1,
                'dir': 1 if self.np_random.random() > 0.5 else -1
            })
        elif type == "orange":
             self.enemies.append({
                'type': 'orange', 'pos': pos, 'color': self.COLOR_ENEMY_ORANGE,
                'speed': self.PLAYER_SPEED * 0.5, 'damage': 2
            })
            
    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['type'] == 'red':
                enemy['pos'][0] += enemy['speed'] * enemy['dir']
                if enemy['pos'][0] > self.WIDTH - self.ENEMY_SIZE/2 or enemy['pos'][0] < self.ENEMY_SIZE/2:
                    enemy['dir'] *= -1
            elif enemy['type'] == 'purple':
                enemy['pos'][1] += enemy['speed'] * enemy['dir']
                if enemy['pos'][1] > self.HEIGHT - self.ENEMY_SIZE/2 or enemy['pos'][1] < self.ENEMY_SIZE/2:
                    enemy['dir'] *= -1
            elif enemy['type'] == 'orange':
                direction_to_player = self.player_pos - enemy['pos']
                dist = np.linalg.norm(direction_to_player)
                if dist > 1:
                    enemy['pos'] += (direction_to_player / dist) * enemy['speed']
    
    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return 0
        distances = [np.linalg.norm(self.player_pos - gem_pos) for gem_pos in self.gems]
        return min(distances)

    def _render_game(self):
        # Render gems with sparkle
        for gem_pos in self.gems:
            pulse = abs(math.sin(self.steps * 0.1 + gem_pos[0])) # Unique sparkle per gem
            radius = int(self.GEM_SIZE / 2 * (1 + 0.2 * pulse))
            glow_radius = int(radius * 1.5)
            glow_alpha = int(100 * pulse)
            
            pygame.gfxdraw.filled_circle(self.screen, int(gem_pos[0]), int(gem_pos[1]), glow_radius, (*self.COLOR_GEM, glow_alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(gem_pos[0]), int(gem_pos[1]), radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, int(gem_pos[0]), int(gem_pos[1]), radius, self.COLOR_GEM)

        # Render enemies
        for enemy in self.enemies:
            self._draw_triangle(self.screen, enemy['color'], enemy['pos'], self.ENEMY_SIZE)
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render player
        is_flashing = self.invulnerability_timer > 0 and (self.steps // 4) % 2 == 0
        if not is_flashing:
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            
    def _draw_triangle(self, surface, color, center_pos, size):
        angle_rad = math.atan2(self.player_pos[1] - center_pos[1], self.player_pos[0] - center_pos[0])
        points = []
        for i in range(3):
            angle = angle_rad + (i * 2 * math.pi / 3)
            x = center_pos[0] + size / 2 * math.cos(angle)
            y = center_pos[1] + size / 2 * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Health Bar
        health_pct = max(0, self.player_health / self.MAX_HEALTH)
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(bar_width * health_pct), bar_height), border_radius=3)

        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if not self.gems else "GAME OVER"
            color = self.COLOR_GEM if not self.gems else self.COLOR_ENEMY_RED
            
            text = self.font_game_over.render(message, True, color)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)
            
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
            })
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['radius'] < 0: p['radius'] = 0

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To actually see the game, you need a different setup
    # This is a basic interactive loop for human play
    
    # Re-initialize with display
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        movement_action = 0 # no-op
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a surface for display
        # Pygame uses (W, H) but obs is (H, W, C), so transpose is needed
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()