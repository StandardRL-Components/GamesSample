
# Generated: 2025-08-27T13:40:12.913843
# Source Brief: brief_00446.md
# Brief Index: 446

        
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
        "Controls: Arrow keys to move the placement cursor. Press space to build a tower at the cursor's location."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of descending enemies by strategically placing defensive towers in this streamlined, top-down tower defense experience."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (25, 30, 45)
    COLOR_PLAYER_BASE = (0, 150, 50)
    COLOR_PLAYER_BASE_GLOW = (0, 200, 100)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_GLOW = (255, 100, 100)
    COLOR_TOWER = (50, 100, 220)
    COLOR_TOWER_GLOW = (100, 150, 255)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (80, 20, 20)
    COLOR_UI_BAR_FILL = (0, 200, 100)
    COLOR_CURSOR_VALID = (100, 255, 100, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100, 100)

    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GAME_WIDTH, GAME_HEIGHT = 200, 200
    FPS = 60
    MAX_STEPS = 30 * FPS  # 30 seconds

    BASE_MAX_HEALTH = 100
    ENEMY_SPAWN_RATE = 2 * FPS  # Every 2 seconds
    ENEMY_SPEED = 1.0
    ENEMY_DAMAGE = 10
    ENEMY_HEALTH = 1

    TOWER_COST = 10
    MAX_TOWERS = 10
    TOWER_RANGE = 100
    TOWER_FIRE_RATE = 1 * FPS # Every 1 second
    TOWER_PROJECTILE_SPEED = 4.0
    TOWER_PROJECTILE_DAMAGE = 1

    INITIAL_RESOURCES = 10
    CURSOR_SPEED = 10


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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game area positioning
        self.game_area_x_offset = (self.SCREEN_WIDTH - self.GAME_WIDTH) // 2
        self.game_area_y_offset = (self.SCREEN_HEIGHT - self.GAME_HEIGHT) // 2
        self.game_rect = pygame.Rect(
            self.game_area_x_offset, self.game_area_y_offset,
            self.GAME_WIDTH, self.GAME_HEIGHT
        )

        # Initialize state variables
        self.cursor_pos = None
        self.base_health = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.victory = None
        self.towers = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.enemy_spawn_timer = None
        self.space_was_held = None
        
        self.reset()
        
        # self.validate_implementation() # For development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = self.INITIAL_RESOURCES
        self.base_health = self.BASE_MAX_HEALTH
        self.game_over = False
        self.victory = False

        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE
        self.space_was_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)

        self._spawn_enemies()
        reward += self._update_towers()
        reward += self._update_projectiles()
        self._update_enemies() # Base damage is a terminal penalty, not a per-step reward
        self._update_particles()

        terminated = False
        if self.base_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
            # Sound effect: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.victory = True
            reward += 100
            # Sound effect: victory.wav
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED

        # Clamp cursor to game area
        self.cursor_pos.x = np.clip(self.cursor_pos.x, self.game_rect.left, self.game_rect.right)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, self.game_rect.top, self.game_rect.bottom)

        # Place tower (on key press, not hold)
        can_place = self.score >= self.TOWER_COST and len(self.towers) < self.MAX_TOWERS
        if space_held and not self.space_was_held and can_place:
            self.towers.append({
                'pos': self.cursor_pos.copy(),
                'cooldown': 0
            })
            self.score -= self.TOWER_COST
            self._create_particles(self.cursor_pos, 20, self.COLOR_TOWER_GLOW, 1.5)
            # Sound effect: build_tower.wav
        
        self.space_was_held = space_held

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE
            spawn_x = self.game_rect.left + self.np_random.random() * self.GAME_WIDTH
            self.enemies.append({
                'pos': pygame.Vector2(spawn_x, self.game_rect.top),
                'health': self.ENEMY_HEALTH
            })
            # Sound effect: enemy_spawn.wav

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = tower['pos'].distance_to(enemy['pos'])
                    if dist <= self.TOWER_RANGE and dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        'pos': tower['pos'].copy(),
                        'target': target,
                        'target_pos': target['pos'].copy() # Cache position in case target dies
                    })
                    tower['cooldown'] = self.TOWER_FIRE_RATE
                    # Sound effect: tower_shoot.wav
        return 0 # No reward for just firing

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] in self.enemies:
                direction = (proj['target']['pos'] - proj['pos']).normalize()
            else: # Target is gone, fly towards its last known position
                if proj['pos'].distance_to(proj['target_pos']) < self.TOWER_PROJECTILE_SPEED:
                    self.projectiles.remove(proj)
                    continue
                direction = (proj['target_pos'] - proj['pos']).normalize()

            proj['pos'] += direction * self.TOWER_PROJECTILE_SPEED

            # Check for collision
            if proj['target'] in self.enemies and proj['pos'].distance_to(proj['target']['pos']) < 5:
                proj['target']['health'] -= self.TOWER_PROJECTILE_DAMAGE
                self._create_particles(proj['pos'], 5, self.COLOR_PROJECTILE, 0.5)
                # Sound effect: projectile_hit.wav
                reward += 0.1
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
            elif not self.game_rect.collidepoint(proj['pos']):
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
        return reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            # Movement
            enemy['pos'].y += self.ENEMY_SPEED
            
            # Check for death
            if enemy['health'] <= 0:
                self.score += 1 # Resource gain
                self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY_GLOW, 2.0)
                # Sound effect: enemy_explode.wav
                self.enemies.remove(enemy)
                # The +1 reward for killing is handled by the gym wrapper/agent logic based on score increase
                continue
            
            # Check for reaching base
            if enemy['pos'].y >= self.game_rect.bottom:
                self.base_health -= self.ENEMY_DAMAGE
                self._create_particles(enemy['pos'], 40, self.COLOR_PLAYER_BASE_GLOW, 3.0)
                # Sound effect: base_damage.wav
                self.enemies.remove(enemy)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.05)
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.random() * 3 + 1
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game_area()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_base()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_area(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.game_rect, 1)

    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            is_ready = tower['cooldown'] == 0
            
            # Glow effect
            glow_color = self.COLOR_TOWER_GLOW if is_ready else (60, 80, 120)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, glow_color)
            
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_TOWER, (pos[0] - 5, pos[1] - 5, 10, 10))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            p1 = (pos[0], pos[1] + 6)
            p2 = (pos[0] - 5, pos[1] - 3)
            p3 = (pos[0] + 5, pos[1] - 3)
            
            # Glow
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'])
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_base(self):
        base_rect = pygame.Rect(self.game_rect.left, self.game_rect.bottom, self.GAME_WIDTH, 5)
        # Glow
        glow_rect = base_rect.inflate(0, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_BASE_GLOW, 50), glow_surf.get_rect(), border_radius=3)
        self.screen.blit(glow_surf, glow_rect.topleft)
        # Main Bar
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_BASE, base_rect, border_radius=3)

    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        can_place = self.score >= self.TOWER_COST and len(self.towers) < self.MAX_TOWERS
        
        # Range indicator
        range_color = self.COLOR_CURSOR_VALID if can_place else self.COLOR_CURSOR_INVALID
        range_surface = pygame.Surface((self.TOWER_RANGE * 2, self.TOWER_RANGE * 2), pygame.SRCALPHA)
        pygame.draw.circle(range_surface, range_color, (self.TOWER_RANGE, self.TOWER_RANGE), self.TOWER_RANGE)
        self.screen.blit(range_surface, (pos[0] - self.TOWER_RANGE, pos[1] - self.TOWER_RANGE), special_flags=pygame.BLEND_RGBA_ADD)

        # Crosshair
        pygame.draw.line(self.screen, (255,255,255), (pos[0] - 8, pos[1]), (pos[0] + 8, pos[1]), 1)
        pygame.draw.line(self.screen, (255,255,255), (pos[0], pos[1] - 8), (pos[0], pos[1] + 8), 1)

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_ratio = max(0, self.base_health / self.BASE_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (10, 10, health_bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"RESOURCES: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 12))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_small.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH/2 - time_text.get_width()/2, 12))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        text = "VICTORY" if self.victory else "GAME OVER"
        color = self.COLOR_PLAYER_BASE_GLOW if self.victory else self.COLOR_ENEMY_GLOW
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    while running:
        # Human controls
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if terminated:
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
        else:
            obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()