
# Generated: 2025-08-27T19:08:08.923514
# Source Brief: brief_02060.md
# Brief Index: 2060

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Alien:
    def __init__(self, pos, alien_type, speed_multiplier, fire_rate_multiplier, np_random):
        self.pos = np.array(pos, dtype=np.float32)
        self.type = alien_type
        self.np_random = np_random
        self.base_speed = 1.5
        self.speed = self.base_speed * speed_multiplier
        self.fire_rate = 0.2 * fire_rate_multiplier  # shots per second
        self.fire_cooldown_max = int(30 / self.fire_rate) if self.fire_rate > 0 else 10000
        self.fire_cooldown = self.np_random.integers(0, self.fire_cooldown_max + 1)
        self.pattern_time = self.np_random.random() * 2 * math.pi
        self.size = 12
        self.glow_size = 20

        if self.type == 'grunt':
            self.color = (255, 64, 64)
        elif self.type == 'dasher':
            self.color = (255, 165, 0)
            self.vel = np.array([self.speed if self.np_random.random() > 0.5 else -self.speed, self.speed * 0.5], dtype=np.float32)
        elif self.type == 'hunter':
            self.color = (64, 255, 64)

    def update(self, player_pos):
        self.pattern_time += 0.05
        if self.type == 'grunt':
            self.pos[0] += math.sin(self.pattern_time) * 1.5
            self.pos[1] += self.speed * 0.20
        elif self.type == 'dasher':
            self.pos += self.vel
            if self.pos[0] < self.size or self.pos[0] > 640 - self.size:
                self.vel[0] *= -1
        elif self.type == 'hunter':
            target_direction = player_pos - self.pos
            norm = np.linalg.norm(target_direction)
            if norm > 0:
                self.pos += (target_direction / norm) * self.speed * 0.4
        
        # Keep aliens on screen for a while
        if self.pos[1] > 400 + self.size:
             self.pos[1] = -self.size

        # Fire logic
        self.fire_cooldown -= 1
        if self.fire_cooldown <= 0:
            self.fire_cooldown = self.fire_cooldown_max
            return True
        return False

class Projectile:
    def __init__(self, pos, vel, color, size):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel

class Particle:
    def __init__(self, pos, np_random):
        self.pos = np.array(pos, dtype=np.float32)
        angle = np_random.random() * 2 * math.pi
        speed = np_random.random() * 3 + 1
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
        self.lifespan = np_random.integers(20, 40)
        self.max_lifespan = self.lifespan
        self.color = random.choice([(255, 255, 0), (255, 165, 0), (255, 255, 255)])
        self.size = np_random.integers(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifespan -= 1

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrow keys to move your ship. Press space to fire your weapon."
    game_description = "A fast-paced, top-down arcade shooter. Survive three waves of alien invaders by destroying them with your laser cannon. Features vibrant neon visuals and particle effects."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SPEED = 6
    PLAYER_FIRE_COOLDOWN = 6  # 5 shots per second
    PLAYER_PROJECTILE_SPEED = 12
    ENEMY_PROJECTILE_SPEED = 5
    MAX_STEPS = 5000
    TOTAL_WAVES = 3

    # --- COLORS (Neon) ---
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_PROJECTILE = (0, 255, 255)
    COLOR_ENEMY_PROJECTILE = (255, 100, 0)
    COLOR_UI_TEXT = (220, 220, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_wave = None
        self.wave_clear_bonus_given = None
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.player_lives = 3
        self.player_fire_cooldown = 0
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 1
        self.wave_clear_bonus_given = False

        self._generate_stars()
        self._spawn_wave(self.current_wave)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Small penalty for time passing
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- UPDATE GAME LOGIC ---
        self._handle_input(movement, space_held)
        self._update_projectiles()
        reward += self._update_aliens()
        self._update_particles()

        # --- HANDLE COLLISIONS ---
        reward += self._handle_collisions()

        # --- CHECK WAVE/GAME STATE ---
        if not self.aliens and not self.wave_clear_bonus_given:
            if self.current_wave < self.TOTAL_WAVES:
                reward += 100  # Wave clear bonus
                self.current_wave += 1
                self._spawn_wave(self.current_wave)
            else: # Game won
                reward += 500
                self.game_over = True
            self.wave_clear_bonus_given = True


        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], 15, self.HEIGHT - 15)
        
        # Player Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if space_held and self.player_fire_cooldown == 0:
            # SFX: Player shoot
            pos = self.player_pos - np.array([0, 20])
            vel = [0, -self.PLAYER_PROJECTILE_SPEED]
            self.player_projectiles.append(Projectile(pos, vel, self.COLOR_PLAYER_PROJECTILE, 4))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN

    def _update_projectiles(self):
        for p in self.player_projectiles: p.update()
        for p in self.enemy_projectiles: p.update()
        
        # Remove off-screen projectiles
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p.pos[1] < self.HEIGHT]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p.pos[1] < self.HEIGHT]

    def _update_aliens(self):
        reward = 0
        for alien in self.aliens:
            if alien.update(self.player_pos):
                # SFX: Enemy shoot
                target_direction = self.player_pos - alien.pos
                norm = np.linalg.norm(target_direction)
                if norm > 0:
                    vel = (target_direction / norm) * self.ENEMY_PROJECTILE_SPEED
                    self.enemy_projectiles.append(Projectile(alien.pos, vel, self.COLOR_ENEMY_PROJECTILE, 3))
        return reward

    def _update_particles(self):
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Aliens
        for p_proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if np.linalg.norm(p_proj.pos - alien.pos) < alien.size + p_proj.size:
                    # SFX: Explosion
                    self._create_explosion(alien.pos)
                    self.aliens.remove(alien)
                    self.player_projectiles.remove(p_proj)
                    self.score += 100
                    reward += 1
                    self.wave_clear_bonus_given = False # Reset bonus flag
                    break
        
        # Enemy projectiles vs Player
        for e_proj in self.enemy_projectiles[:]:
            if np.linalg.norm(e_proj.pos - self.player_pos) < 10 + e_proj.size:
                # SFX: Player hit
                self.enemy_projectiles.remove(e_proj)
                self._create_explosion(self.player_pos)
                self.player_lives -= 1
                reward -= 1
                if self.player_lives <= 0:
                    reward -= 100 # Game over penalty
                    self.game_over = True
                break
        return reward

    def _spawn_wave(self, wave_num):
        speed_mult = 1 + (wave_num - 1) * 0.05
        fire_rate_mult = 1 + ((wave_num - 1) * 0.05) / 0.2

        if wave_num == 1:
            for i in range(8):
                x = self.WIDTH/2 + (i-3.5) * 60
                y = 60 + abs(i-3.5) * 20
                self.aliens.append(Alien((x, y), 'grunt', speed_mult, fire_rate_mult, self.np_random))
        elif wave_num == 2:
            for i in range(6): # Grunts
                self.aliens.append(Alien((80 + i*80, 60), 'grunt', speed_mult, fire_rate_mult, self.np_random))
            for i in range(4): # Dashers
                self.aliens.append(Alien((120 + i*100, 120), 'dasher', speed_mult, fire_rate_mult, self.np_random))
        elif wave_num == 3:
            for i in range(5): # Grunts
                self.aliens.append(Alien((80 + i*100, 60), 'grunt', speed_mult, fire_rate_mult, self.np_random))
            for i in range(4): # Dashers
                self.aliens.append(Alien((120 + i*100, 120), 'dasher', speed_mult, fire_rate_mult, self.np_random))
            for i in range(2): # Hunters
                self.aliens.append(Alien((self.WIDTH/2 - 100 + i*200, 180), 'hunter', speed_mult, fire_rate_mult, self.np_random))

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT],
                'speed': self.np_random.random() * 0.5 + 0.2,
                'size': self.np_random.integers(1, 3),
                'color': random.choice([(100,100,150), (150,150,200), (200,200,255)])
            })

    def _create_explosion(self, pos, num_particles=30):
        for _ in range(num_particles):
            self.particles.append(Particle(pos, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives, "wave": self.current_wave}

    def _render_game(self):
        # Stars
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT: star['pos'][1] = 0
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'][0]), int(star['pos'][1])), star['size'])
        
        # Effects surface for glow
        effects_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        # Player
        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            glow_color = (*self.COLOR_PLAYER, 80)
            pygame.gfxdraw.filled_circle(effects_surface, px, py, 18, glow_color)
            pygame.gfxdraw.aacircle(effects_surface, px, py, 18, glow_color)
            
            # Ship body
            ship_points = [(px, py - 15), (px - 10, py + 10), (px + 10, py + 10)]
            pygame.gfxdraw.aapolygon(effects_surface, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(effects_surface, ship_points, self.COLOR_PLAYER)
        
        # Aliens
        for alien in self.aliens:
            ax, ay = int(alien.pos[0]), int(alien.pos[1])
            glow_color = (*alien.color, 100)
            pygame.gfxdraw.filled_circle(effects_surface, ax, ay, alien.glow_size, glow_color)
            pygame.gfxdraw.aacircle(effects_surface, ax, ay, alien.glow_size, glow_color)
            pygame.gfxdraw.filled_circle(effects_surface, ax, ay, alien.size, alien.color)
            pygame.gfxdraw.aacircle(effects_surface, ax, ay, alien.size, alien.color)

        # Projectiles
        for p in self.player_projectiles:
            px, py = int(p.pos[0]), int(p.pos[1])
            glow_color = (*p.color, 120)
            pygame.draw.line(effects_surface, glow_color, (px, py), (px, py + 10), p.size + 4)
            pygame.draw.line(effects_surface, p.color, (px, py), (px, py + 5), p.size)

        for p in self.enemy_projectiles:
            px, py = int(p.pos[0]), int(p.pos[1])
            glow_color = (*p.color, 120)
            pygame.gfxdraw.filled_circle(effects_surface, px, py, p.size + 3, glow_color)
            pygame.gfxdraw.filled_circle(effects_surface, px, py, p.size, p.color)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(effects_surface, int(p.pos[0]), int(p.pos[1]), int(p.size), color)

        self.screen.blit(effects_surface, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (10, 10))
        for i in range(self.player_lives):
            px, py = 80 + i * 25, 18
            ship_points = [(px, py - 8), (px - 5, py + 5), (px + 5, py + 5)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_points)

        # Wave
        wave_text = self.font_large.render(f"WAVE {self.current_wave}", True, self.COLOR_UI_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH/2, 30))
        self.screen.blit(wave_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            if self.player_lives <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            end_text = self.font_large.render(msg, True, (255, 50, 50) if self.player_lives <= 0 else (50, 255, 50))
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize state for _get_observation
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset(seed=0)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    # --- For interactive play ---
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause before quitting

        clock.tick(30) # Run at 30 FPS

    env.close()