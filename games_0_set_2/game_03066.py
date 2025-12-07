
# Generated: 2025-08-27T22:15:53.554837
# Source Brief: brief_03066.md
# Brief Index: 3066

        
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
        "A retro arcade shooter. Survive five waves of descending aliens by shooting them down while dodging their fire."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Sizing
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = (20, 15)
    ALIEN_SIZE = 18
    PROJ_RADIUS = 3

    # Colors (Bright on Dark)
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 40)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_ALIEN_ANGLED = (255, 150, 50)
    COLOR_PROJECTILE_PLAYER = (200, 200, 255)
    COLOR_PROJECTILE_ENEMY = (255, 200, 200)
    COLOR_EXPLOSION = (255, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_WAVE_CLEAR = (100, 255, 100)

    # Game Mechanics
    PLAYER_SPEED = 6
    PLAYER_FIRE_COOLDOWN = 8  # frames
    PROJECTILE_SPEED = 10
    TOTAL_WAVES = 5
    MAX_EPISODE_STEPS = 10000
    PLAYER_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables to be populated in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.current_wave = 0
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.alien_move_direction = 1
        self.alien_move_down_timer = 0
        self.wave_clear_timer = 0
        self.alien_base_speed = 0
        self.alien_fire_rate = 0

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float32)
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_cooldown = 0
        self.current_wave = 1
        
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.alien_move_direction = 1
        self.alien_move_down_timer = 0
        self.wave_clear_timer = 0
        
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.1  # Survival reward per frame
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            terminated = self._check_termination()

        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1

        # Player Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE[0] / 2, self.WIDTH - self.PLAYER_SIZE[0] / 2)

        # Player Firing
        if space_held and self.player_fire_cooldown <= 0:
            # sfx: player_shoot
            self.player_projectiles.append(self.player_pos.copy())
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
            reward = -0.5
            return reward
        return 0

    def _update_game_state(self):
        reward = 0
        
        # Update timers
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        if self.wave_clear_timer > 0:
            self.wave_clear_timer -= 1

        # Update player projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]
        for p in self.player_projectiles:
            p[1] -= self.PROJECTILE_SPEED

        # Update enemy projectiles
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p['pos'][1] < self.HEIGHT]
        for p in self.enemy_projectiles:
            p['pos'] += p['vel']

        # Update aliens
        self._move_aliens()
        self._aliens_fire()
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Collision detection
        reward += self._check_collisions()
        
        # Check for wave clear
        if not self.aliens and not self.game_over:
            # sfx: wave_clear_success
            reward += 100
            self.score += 100 # Wave clear bonus
            self.current_wave += 1
            if self.current_wave <= self.TOTAL_WAVES:
                self._spawn_wave()
                self.wave_clear_timer = 60 # 2 seconds at 30fps
            else:
                self.game_over = True # Player wins
        
        return reward
        
    def _move_aliens(self):
        if not self.aliens:
            return

        move_down = False
        for alien in self.aliens:
            if (alien['pos'][0] >= self.WIDTH - self.ALIEN_SIZE and self.alien_move_direction > 0) or \
               (alien['pos'][0] <= self.ALIEN_SIZE and self.alien_move_direction < 0):
                move_down = True
                break
        
        if move_down:
            self.alien_move_direction *= -1
            for alien in self.aliens:
                alien['pos'][1] += self.ALIEN_SIZE / 2
        else:
            for alien in self.aliens:
                alien['pos'][0] += self.alien_move_direction * self.alien_base_speed

    def _aliens_fire(self):
        for alien in self.aliens:
            if self.np_random.random() < self.alien_fire_rate:
                # sfx: enemy_shoot
                if alien['type'] == 'straight':
                    vel = np.array([0, self.PROJECTILE_SPEED * 0.7], dtype=np.float32)
                else: # angled
                    angle = self.np_random.uniform(-0.5, 0.5)
                    vel = np.array([angle, 1], dtype=np.float32)
                    vel = vel / np.linalg.norm(vel) * self.PROJECTILE_SPEED * 0.7
                
                self.enemy_projectiles.append({'pos': alien['pos'].copy(), 'vel': vel})

    def _check_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        projectiles_to_remove = []
        aliens_to_remove = []
        for i, proj in enumerate(self.player_projectiles):
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                dist = np.linalg.norm(proj - alien['pos'])
                if dist < self.ALIEN_SIZE / 2 + self.PROJ_RADIUS:
                    # sfx: alien_explosion
                    aliens_to_remove.append(j)
                    projectiles_to_remove.append(i)
                    self._create_explosion(alien['pos'], 20, self.COLOR_EXPLOSION)
                    reward += 1
                    self.score += 1
                    break
        
        self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in projectiles_to_remove]
        self.aliens = [a for i, a in enumerate(self.aliens) if i not in aliens_to_remove]
        
        # Enemy projectiles vs Player
        projectiles_to_remove = []
        for i, proj in enumerate(self.enemy_projectiles):
            dist = np.linalg.norm(proj['pos'] - self.player_pos)
            if dist < self.PLAYER_SIZE[0] / 2 + self.PROJ_RADIUS:
                projectiles_to_remove.append(i)
                self._player_hit()
                reward -= 100
                break
        
        self.enemy_projectiles = [p for i, p in enumerate(self.enemy_projectiles) if i not in projectiles_to_remove]

        # Aliens vs Player (if they reach bottom)
        for alien in self.aliens:
            if alien['pos'][1] > self.HEIGHT - 60:
                self._player_hit()
                reward -= 100
                # Remove all aliens to start next attempt/end game
                self.aliens.clear()
                break

        return reward

    def _player_hit(self):
        # sfx: player_explosion
        self.player_lives -= 1
        self._create_explosion(self.player_pos, 40, self.COLOR_PLAYER)
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float32)
        if self.player_lives <= 0:
            self.game_over = True

    def _spawn_wave(self):
        self.aliens.clear()
        self.alien_base_speed = 1.0 + (self.current_wave - 1) * 0.2
        self.alien_fire_rate = 0.001 + (self.current_wave - 1) * 0.001
        
        rows = 3 + min(2, self.current_wave - 1)
        cols = 8
        x_spacing = self.WIDTH * 0.8 / cols
        y_spacing = 40
        start_x = self.WIDTH * 0.1
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                alien_type = self.np_random.choice(['straight', 'angled'], p=[0.7, 0.3])
                pos = np.array([start_x + c * x_spacing, start_y + r * y_spacing], dtype=np.float32)
                self.aliens.append({'pos': pos, 'type': alien_type})
    
    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _check_termination(self):
        return self.player_lives <= 0 or (self.current_wave > self.TOTAL_WAVES and not self.aliens)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, p['color'] + (alpha,))
        
        # Render enemy projectiles
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJ_RADIUS, self.COLOR_PROJECTILE_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJ_RADIUS, self.COLOR_PROJECTILE_ENEMY)

        # Render player projectiles
        for p in self.player_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), self.PROJ_RADIUS, self.COLOR_PROJECTILE_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), self.PROJ_RADIUS, self.COLOR_PROJECTILE_PLAYER)

        # Render aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            size = self.ALIEN_SIZE
            color = self.COLOR_ALIEN if alien['type'] == 'straight' else self.COLOR_ALIEN_ANGLED
            rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        # Render player
        if self.player_lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            w, h = self.PLAYER_SIZE
            points = [(px, py - h // 2), (px - w // 2, py + h // 2), (px + w // 2, py + h // 2)]
            
            # Glow effect
            glow_points = [(px, py - h // 2 - 2), (px - w // 2 - 4, py + h // 2 + 4), (px + w // 2 + 4, py + h // 2 + 4)]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
            
            # Ship body
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            px, py = 20 + i * (self.PLAYER_SIZE[0] + 5), self.HEIGHT - 20
            w, h = self.PLAYER_SIZE
            points = [(px, py - h // 2), (px - w // 2, py + h // 2), (px + w // 2, py + h // 2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            
        # Wave Clear Message
        if self.wave_clear_timer > 0:
            alpha = min(255, int(255 * (self.wave_clear_timer / 30.0)))
            text_surf = self.font_large.render("WAVE CLEARED", True, self.COLOR_WAVE_CLEAR)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

        # Game Over / Win Message
        if self.game_over:
            if self.current_wave > self.TOTAL_WAVES:
                message = "YOU WIN!"
                color = self.COLOR_WAVE_CLEAR
            else:
                message = "GAME OVER"
                color = self.COLOR_ALIEN
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Invaders Clone")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # Action defaults
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    env.close()