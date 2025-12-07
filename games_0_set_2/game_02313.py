
# Generated: 2025-08-28T04:25:05.286218
# Source Brief: brief_02313.md
# Brief Index: 2313

        
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
        "Controls: Use arrow keys to move. Hold Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down shooter. Destroy waves of descending aliens before they overwhelm you."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_PROJECTILE = (255, 255, 255)
        self.COLOR_ALIEN_1 = (255, 80, 80)
        self.COLOR_ALIEN_2 = (255, 160, 80)
        self.COLOR_ALIEN_PROJECTILE = (255, 50, 50)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 50)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)

        # Fonts
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)

        # Game constants
        self.PLAYER_SPEED = 6
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_FIRE_COOLDOWN = 5  # 6 shots per second at 30fps
        self.PLAYER_PROJECTILE_SPEED = 12
        self.MAX_STEPS = 5000
        self.TOTAL_ALIENS_PER_WAVE = 50

        # Base difficulty
        self.BASE_ALIEN_SPEED = 0.5
        self.BASE_ALIEN_FIRE_RATE = 0.002
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_fire_timer = 0
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.aliens_destroyed_total = 0
        self.alien_descent_speed = 0
        self.alien_fire_rate = 0
        self.wave_number = 1
        self.aliens_in_wave_destroyed = 0
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        self.aliens_destroyed_total = 0
        self.aliens_in_wave_destroyed = 0

        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_timer = 0
        
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        
        self._reset_difficulty()
        self._spawn_wave()
        
        # Create a static starfield for the background
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()

    def _reset_difficulty(self):
        self.alien_descent_speed = self.BASE_ALIEN_SPEED
        self.alien_fire_rate = self.BASE_ALIEN_FIRE_RATE

    def _spawn_wave(self):
        self.aliens.clear()
        rows = 5
        cols = 10
        x_spacing = self.WIDTH * 0.8 / cols
        y_spacing = 40
        start_x = self.WIDTH * 0.1
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                alien_type = 1 if r < 2 else 2 # Two types of aliens
                self.aliens.append({
                    "pos": [start_x + c * x_spacing, start_y + r * y_spacing],
                    "type": alien_type,
                    "move_dir": 1, # 1 for right, -1 for left
                })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small survival reward per frame

        self._handle_input(action)
        self._update_player_projectiles()
        self._update_aliens()
        self._update_alien_projectiles()
        self._update_particles()
        
        # Collision checks and rewards
        reward += self._check_collisions()

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self._create_explosion(self.player_pos, 50, self.COLOR_EXPLOSION)
        elif self.aliens_in_wave_destroyed >= self.TOTAL_ALIENS_PER_WAVE:
            reward += 100
            terminated = True # For this brief, one wave is one episode
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right
        
        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], self.HEIGHT - 100, self.HEIGHT - 20)
        
        # Player Firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
            
        if space_held and self.player_fire_timer == 0:
            # SFX: Player shoot
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player_projectiles(self):
        for p in self.player_projectiles:
            p[1] -= self.PLAYER_PROJECTILE_SPEED
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]

    def _update_aliens(self):
        move_sideways = (self.steps // 20) % 2 == 0 # Sideways movement every 20 frames
        
        for alien in self.aliens:
            # Vertical movement
            alien['pos'][1] += self.alien_descent_speed
            
            # Sideways movement
            if move_sideways:
                alien['pos'][0] += 5 * alien['move_dir']

            # Check for wall collision to reverse direction
            if alien['pos'][0] > self.WIDTH - 20 or alien['pos'][0] < 20:
                alien['move_dir'] *= -1

            # Firing logic
            if self.np_random.random() < self.alien_fire_rate:
                # SFX: Alien shoot
                self.alien_projectiles.append(list(alien['pos']))

    def _update_alien_projectiles(self):
        for p in self.alien_projectiles:
            p[1] += 4 # Slower than player projectiles
        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['growth']
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        projectiles_to_remove = []
        aliens_to_remove = []
        
        for i, proj in enumerate(self.player_projectiles):
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                dist = math.hypot(proj[0] - alien['pos'][0], proj[1] - alien['pos'][1])
                if dist < 15: # Collision radius
                    # SFX: Explosion
                    self._create_explosion(alien['pos'], 20, self.COLOR_EXPLOSION)
                    aliens_to_remove.append(j)
                    if i not in projectiles_to_remove:
                        projectiles_to_remove.append(i)
                    self.score += 10
                    reward += 1
                    self.aliens_destroyed_total += 1
                    self.aliens_in_wave_destroyed += 1
                    
                    # Difficulty progression
                    if self.aliens_destroyed_total % 10 == 0:
                        self.alien_descent_speed += 0.05
                    if self.aliens_destroyed_total % 5 == 0:
                        self.alien_fire_rate += 0.001
        
        # Remove collided objects (in reverse order to avoid index errors)
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.player_projectiles[i]
        for i in sorted(aliens_to_remove, reverse=True):
            del self.aliens[i]
            
        # Alien projectiles vs Player
        projectiles_to_remove.clear()
        player_hitbox = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        
        for i, proj in enumerate(self.alien_projectiles):
            if player_hitbox.collidepoint(proj[0], proj[1]):
                # SFX: Player hit
                self.player_health -= 1
                reward -= 1
                self._create_explosion(self.player_pos, 15, [self.COLOR_PLAYER])
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
        
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.alien_projectiles[i]
            
        return reward

    def _create_explosion(self, pos, count, colors):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.random() * 2 + 1,
                "growth": -0.05,
                "color": random.choice(colors),
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_projectiles()
        self._render_aliens()
        if not (self.game_over and self.player_health <= 0):
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "wave": self.wave_number,
            "aliens_left": len(self.aliens)
        }
        
    def _render_background(self):
        for x, y, size in self.stars:
            color_val = 50 + size * 20
            pygame.draw.rect(self.screen, (color_val, color_val, color_val+10), (x, y, size, size))

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        # Ship body
        points = [(x, y - 10), (x - 8, y + 8), (x + 8, y + 8)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        # Engine glow
        glow_color = (200, 255, 255)
        engine_y = y + 10
        engine_width = 4 + (self.player_fire_timer / self.PLAYER_FIRE_COOLDOWN) * 4 # Pulsate
        pygame.draw.rect(self.screen, glow_color, (x - engine_width/2, engine_y, engine_width, 4))


    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            if alien['type'] == 1:
                color = self.COLOR_ALIEN_1
                pygame.gfxdraw.aacircle(self.screen, x, y, 8, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, 8, color)
                pygame.draw.rect(self.screen, color, (x-12, y-4, 24, 8))
            else:
                color = self.COLOR_ALIEN_2
                points = [(x, y-8), (x-8, y+8), (x+8, y+8)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles:
            start = (int(p[0]), int(p[1]))
            end = (int(p[0]), int(p[1] + 8))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJECTILE, start, end, 3)
            
        # Alien projectiles
        for p in self.alien_projectiles:
            x, y = int(p[0]), int(p[1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_ALIEN_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_ALIEN_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            pos = [p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1]]
            p['pos'] = pos
            radius = max(0, p['radius'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_title.render(f"WAVE {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))
        
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 100
        bar_height = 15
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")