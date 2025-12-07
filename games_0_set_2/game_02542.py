
# Generated: 2025-08-27T20:41:21.516589
# Source Brief: brief_02542.md
# Brief Index: 2542

        
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
        "Controls: ←→ to rotate the turret. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemies by strategically rotating and firing a central turret."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500 # Approx 50 seconds at 30 FPS
    TOTAL_WAVES = 5

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_BASE = (60, 180, 70)
    COLOR_BASE_STROKE = (80, 220, 90)
    COLOR_TURRET_BASE = (100, 110, 120)
    COLOR_TURRET_BARREL = (230, 230, 230)
    COLOR_ENEMY = (210, 60, 60)
    COLOR_ENEMY_STROKE = (255, 90, 90)
    COLOR_PROJECTILE = (255, 200, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    
    # Game Parameters
    BASE_RADIUS = 30
    TURRET_ROTATION_SPEED = math.radians(5)
    PROJECTILE_SPEED = 12
    PROJECTILE_RADIUS = 4
    FIRE_COOLDOWN_FRAMES = 6
    ENEMY_SIZE = 20
    
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
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.render_mode = render_mode
        
        # Initialize state variables
        self.turret_angle = 0.0
        self.base_health = 0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.wave_number = 0
        self.score = 0
        self.steps = 0
        self.fire_cooldown = 0
        self.game_over = False
        self.game_won = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.turret_angle = -math.pi / 2  # Point up
        self.base_health = 100
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.fire_cooldown = 0
        self.wave_number = 0
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1

        # 3=left, 4=right
        if movement == 3:
            self.turret_angle -= self.TURRET_ROTATION_SPEED
        elif movement == 4:
            self.turret_angle += self.TURRET_ROTATION_SPEED

        if space_held and self.fire_cooldown <= 0:
            self._fire_projectile()
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
            # Small penalty for firing, offset by hitting
            reward -= 0.01

        # --- Game Logic Update ---
        self.steps += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Wave Management ---
        if not self.enemies and self.enemies_spawned_this_wave >= self.enemies_in_wave:
            if self.wave_number >= self.TOTAL_WAVES:
                self.game_won = True
                self.game_over = True
            else:
                self._start_next_wave()
        
        # --- Termination Check ---
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            terminated = True
            reward -= 100  # Large penalty for losing
            self._create_explosion(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, 100, 50)
        
        if self.game_won:
            terminated = True
            reward += 100 # Large reward for winning

        if self.steps >= self.MAX_STEPS and not self.game_won:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_next_wave(self):
        self.wave_number += 1
        self.enemies_in_wave = 5 + self.wave_number * 3
        self.enemies_spawned_this_wave = 0
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = max(10, 60 - self.wave_number * 5)
        self.enemy_speed = 0.5 + self.wave_number * 0.1
        self.enemy_health = 1 + (self.wave_number - 1) // 2

    def _fire_projectile(self):
        # sfx: Laser_Shoot
        start_x = self.SCREEN_WIDTH / 2 + math.cos(self.turret_angle) * (self.BASE_RADIUS + 5)
        start_y = self.SCREEN_HEIGHT / 2 + math.sin(self.turret_angle) * (self.BASE_RADIUS + 5)
        
        projectile = {
            "x": start_x, "y": start_y,
            "vx": math.cos(self.turret_angle) * self.PROJECTILE_SPEED,
            "vy": math.sin(self.turret_angle) * self.PROJECTILE_SPEED
        }
        self.projectiles.append(projectile)
        # Muzzle flash
        self._create_explosion(start_x, start_y, 10, 3, self.COLOR_PROJECTILE)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            hit_enemy = False
            for e in self.enemies:
                if pygame.Rect(p['x']-self.PROJECTILE_RADIUS, p['y']-self.PROJECTILE_RADIUS, 
                               self.PROJECTILE_RADIUS*2, self.PROJECTILE_RADIUS*2).colliderect(e['rect']):
                    # sfx: Hit_Hurt
                    e['health'] -= 1
                    hit_enemy = True
                    self._create_explosion(p['x'], p['y'], 15, 5)
                    reward += 0.1 # Reward for hitting
                    if e['health'] <= 0:
                        e['alive'] = False
                        self.score += 10
                        reward += 1 # Reward for destroying
                        self._create_explosion(e['rect'].centerx, e['rect'].centery, 30, 10)
                    break
            
            if not hit_enemy and 0 < p['x'] < self.SCREEN_WIDTH and 0 < p['y'] < self.SCREEN_HEIGHT:
                projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        self.enemies = [e for e in self.enemies if e.get('alive', True)]
        return reward

    def _update_enemies(self):
        reward = 0
        # Spawn new enemies for the wave
        self.enemy_spawn_timer += 1
        if self.enemies_spawned_this_wave < self.enemies_in_wave and self.enemy_spawn_timer > self.enemy_spawn_interval:
            self._spawn_enemy()
            self.enemy_spawn_timer = 0
            self.enemies_spawned_this_wave += 1

        # Move existing enemies
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        for e in self.enemies:
            dx = center_x - e['rect'].centerx
            dy = center_y - e['rect'].centery
            dist = math.hypot(dx, dy)
            
            if dist < self.BASE_RADIUS + self.ENEMY_SIZE / 2:
                # sfx: Explosion
                self.base_health -= 10
                e['alive'] = False
                self._create_explosion(e['rect'].centerx, e['rect'].centery, 40, 15)
                continue

            if dist > 1:
                e['rect'].x += (dx / dist) * self.enemy_speed
                e['rect'].y += (dy / dist) * self.enemy_speed

        self.enemies = [e for e in self.enemies if e.get('alive', True)]
        return reward

    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: # Top
            x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE
        elif side == 1: # Right
            x, y = self.SCREEN_WIDTH + self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif side == 2: # Bottom
            x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SIZE
        else: # Left
            x, y = -self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        
        enemy = {
            "rect": pygame.Rect(x, y, self.ENEMY_SIZE, self.ENEMY_SIZE),
            "health": self.enemy_health,
            "max_health": self.enemy_health,
        }
        self.enemies.append(enemy)

    def _create_explosion(self, x, y, max_radius, num_particles, color=None):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            start_radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            particle_color = color if color is not None else (
                self.np_random.integers(200, 256), self.np_random.integers(100, 200), 0
            )
            self.particles.append({
                "x": x, "y": y, "vx": vx, "vy": vy,
                "radius": start_radius, "lifetime": lifetime, "max_lifetime": lifetime,
                "color": particle_color
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2

        # Draw Base
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.BASE_RADIUS, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.BASE_RADIUS, self.COLOR_BASE_STROKE)

        # Draw Enemies
        for e in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_STROKE, e['rect'], 2, border_radius=3)
            # Enemy health bar
            if e['health'] < e['max_health']:
                health_pct = e['health'] / e['max_health']
                bar_width = self.ENEMY_SIZE * health_pct
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (e['rect'].x, e['rect'].y - 7, self.ENEMY_SIZE, 4))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (e['rect'].x, e['rect'].y - 7, bar_width, 4))

        # Draw Projectiles
        for p in self.projectiles:
            px, py = int(p['x']), int(p['y'])
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Draw Turret
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 10, self.COLOR_TURRET_BASE)
        barrel_end_x = center_x + math.cos(self.turret_angle) * (self.BASE_RADIUS)
        barrel_end_y = center_y + math.sin(self.turret_angle) * (self.BASE_RADIUS)
        pygame.draw.line(self.screen, self.COLOR_TURRET_BARREL, (center_x, center_y), (barrel_end_x, barrel_end_y), 6)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['x'] - p['radius'], p['y'] - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Base Health Bar
        health_pct = self.base_health / 100
        bar_width = 100
        bar_height = 15
        bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=3)

        # Game Over / Win Message
        if self.game_over:
            if self.game_won:
                msg = "YOU WIN!"
                color = self.COLOR_BASE
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Turret Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()