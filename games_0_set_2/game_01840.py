
# Generated: 2025-08-27T18:28:56.181890
# Source Brief: brief_01840.md
# Brief Index: 1840

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from waves of descending alien invaders in a visually stunning side-view shooter."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_BULLET = (100, 200, 255)
        self.COLOR_ALIEN_TYPES = {
            'slow': (200, 200, 200),
            'medium': (255, 150, 255),
            'fast': (255, 100, 100),
        }
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (50, 205, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_EXPLOSION = [(255, 255, 0), (255, 165, 0), (255, 255, 255)]

        # Game constants
        self.MAX_STEPS = 2000
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 8
        self.PLAYER_BULLET_SPEED = 12
        self.ALIEN_BULLET_SPEEDS = {'slow': 4, 'medium': 6, 'fast': 8}
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_fire_timer = 0
        self.wave = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_health = 100
        self.player_fire_timer = 0
        
        self.wave = 1
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self._spawn_stars()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'size': self.np_random.integers(1, 3),
                'speed': self.np_random.random() * 0.5 + 0.25,
                'color': random.choice([(100,100,100), (150,150,150), (200,200,200)])
            })
            
    def _spawn_wave(self):
        self.aliens = []
        num_aliens = 10 + (self.wave - 1) * 2
        alien_types = ['slow', 'medium', 'fast']
        
        rows = math.ceil(num_aliens / 10)
        cols = 10
        
        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            
            alien_type = self.np_random.choice(alien_types)
            fire_rate_increase = min(0.9, (self.wave - 1) * 0.05) # Cap at 1 shot/sec
            
            self.aliens.append({
                'pos': [col * 50 + 75, -50 - row * 50],
                'type': alien_type,
                'fire_cooldown': self.np_random.integers(60, 180),
                'base_fire_cooldown': int(60 * (1.0 - fire_rate_increase)),
                'sine_offset': self.np_random.random() * 2 * math.pi,
                'sine_amplitude': self.np_random.integers(20, 50),
            })
            
    def _create_explosion(self, pos, num_particles=20):
        # Sound effect placeholder: # sfx_explosion.play()
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
            
        if space_held and self.player_fire_timer == 0:
            # Sound effect placeholder: # sfx_player_shoot.play()
            self.player_projectiles.append({'pos': [self.player_pos[0], self.player_pos[1] - 20]})
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

        # --- Update Game State ---
        self.steps += 1
        
        # Update stars
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'] = [self.np_random.integers(0, self.WIDTH), -star['size']]

        # Update player projectiles
        for proj in self.player_projectiles[:]:
            proj['pos'][1] -= self.PLAYER_BULLET_SPEED
            if proj['pos'][1] < 0:
                self.player_projectiles.remove(proj)

        # Update alien projectiles
        for proj in self.alien_projectiles[:]:
            proj['pos'][1] += self.ALIEN_BULLET_SPEEDS[proj['type']]
            if proj['pos'][1] > self.HEIGHT:
                self.alien_projectiles.remove(proj)
                
        # Update aliens
        for alien in self.aliens:
            alien['pos'][1] += 0.75 # Vertical descent
            alien['pos'][0] += math.sin(self.steps * 0.05 + alien['sine_offset']) * 0.5
            
            # Alien firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0 and alien['pos'][1] > 0:
                # Sound effect placeholder: # sfx_alien_shoot.play()
                self.alien_projectiles.append({
                    'pos': list(alien['pos']),
                    'type': alien['type']
                })
                alien['fire_cooldown'] = self.np_random.integers(alien['base_fire_cooldown'], alien['base_fire_cooldown'] * 2)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 10, 30, 20)
        
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - 2, proj['pos'][1] - 8, 4, 16)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0] - 15, alien['pos'][1] - 10, 30, 20)
                if proj_rect.colliderect(alien_rect):
                    reward += 0.5  # Reward for hitting
                    reward += 10   # Reward for destroying
                    self.score += 10
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    break 

        # Alien projectiles vs Player
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - 4, proj['pos'][1] - 4, 8, 8)
            if player_rect.colliderect(proj_rect):
                reward -= 2.0  # Penalty for being hit
                self.player_health -= 10
                self._create_explosion([self.player_pos[0], self.player_pos[1]], 10)
                self.alien_projectiles.remove(proj)
                # Sound effect placeholder: # sfx_player_hit.play()

        # Aliens vs Player
        for alien in self.aliens[:]:
            alien_rect = pygame.Rect(alien['pos'][0] - 15, alien['pos'][1] - 10, 30, 20)
            if player_rect.colliderect(alien_rect):
                self.player_health = 0 # Instant death on collision

        # --- Check for wave clear ---
        if not self.aliens and not self.game_over:
            reward += 50  # Reward for clearing wave
            self.wave += 1
            self.score += 100
            self._spawn_wave()
            
        # --- Termination Check ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            if self.player_health <= 0:
                reward -= 20 # Penalty for dying
                self._create_explosion(self.player_pos, 40)
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'][0]), int(star['pos'][1])), star['size'])
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((p['lifespan'], p['lifespan']), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['lifespan']//2, p['lifespan']//2), p['lifespan']//2)
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['lifespan']//2), int(p['pos'][1] - p['lifespan']//2)))

        # Render aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            color = self.COLOR_ALIEN_TYPES[alien['type']]
            points = [(x, y - 12), (x - 15, y + 10), (x + 15, y + 10)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render player projectiles
        for proj in self.player_projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            # Trail effect
            for i in range(4):
                alpha = 255 - i * 60
                pygame.gfxdraw.filled_circle(self.screen, x, y + i * 5, 3-i//2, (*self.COLOR_PLAYER_BULLET, alpha))
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_PLAYER_BULLET)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_PLAYER_BULLET)

        # Render alien projectiles
        for proj in self.alien_projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            color = self.COLOR_ALIEN_TYPES[proj['type']]
            pygame.gfxdraw.aacircle(self.screen, x, y, 5, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 5, color)
        
        # Render player
        if self.player_health > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            points = [(x, y - 15), (x - 18, y + 12), (x + 18, y + 12)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            # Engine glow
            glow_y = y + 12
            glow_h = max(0, min(15, self.np_random.integers(5, 15)))
            pygame.draw.rect(self.screen, (255, 255, 200), (x-5, glow_y, 10, glow_h), border_radius=3)
        
        # Render UI
        # Health bar
        health_pct = max(0, self.player_health / 100.0)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_pct), 20))
        
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
        
        # Wave
        wave_surf = self.font_large.render(f"WAVE {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.WIDTH // 2 - wave_surf.get_width() // 2, self.HEIGHT - 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            game_over_surf = self.font_large.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(game_over_surf, (self.WIDTH // 2 - game_over_surf.get_width() // 2, self.HEIGHT // 2 - 30))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
    }

    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Defender")
    clock = pygame.time.Clock()

    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Action Mapping ---
        movement = 0 # No-op
        if keys_held[pygame.K_LEFT]:
            movement = 3
        elif keys_held[pygame.K_RIGHT]:
            movement = 4
            
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 0 # Not used

        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Frame Rate ---
        clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
    pygame.time.wait(2000)
    env.close()