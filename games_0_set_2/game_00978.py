
# Generated: 2025-08-27T15:23:40.042473
# Source Brief: brief_00978.md
# Brief Index: 978

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade shooter. Defend against 5 waves of descending aliens. Survive and get the high score!"
    )

    # Frames auto-advance for smooth, real-time gameplay.
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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ALIEN = (255, 50, 100)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (200, 0, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (200, 50, 0)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_WIN_TEXT = (100, 255, 100)
        self.COLOR_LOSE_TEXT = (255, 100, 100)

        # Game constants
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 5  # frames
        self.PLAYER_PROJ_SPEED = 10
        self.ENEMY_PROJ_SPEED = 5
        self.MAX_WAVES = 5
        self.MAX_STEPS = 10000
        self.WAVE_CLEAR_DELAY = 90  # 3 seconds at 30fps

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_timer = 0
        self.current_wave = 0
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        self.wave_clear_timer = 0
        self.win_state = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.player_fire_timer = 0
        self.current_wave = 1
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.wave_clear_timer = 0

        self._create_stars()
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            self._handle_input(action)
            
            if self.wave_clear_timer > 0:
                self.wave_clear_timer -= 1
                if self.wave_clear_timer == 0:
                    self.current_wave += 1
                    if self.current_wave > self.MAX_WAVES:
                        self.game_over = True
                        self.win_state = True
                        reward += 100.0 # Win game
                    else:
                        self._spawn_wave()
            else:
                self._update_player()
                self._update_aliens()
                self._update_projectiles()
                
                collision_rewards = self._handle_collisions()
                reward += collision_rewards
            
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True # Termination due to step limit
            if not self.win_state:
                 reward -= 100.0 # Lose game due to timeout

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.HEIGHT - 20)

        # Firing
        if space_held and self.player_fire_timer == 0:
            self._fire_player_projectile()
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            # sfx: player_shoot

    def _update_player(self):
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1

    def _update_aliens(self):
        alien_speed = 0.5 + (self.current_wave - 1) * 0.05
        alien_fire_rate = 0.01 + (self.current_wave - 1) * 0.001
        
        for alien in self.aliens:
            # Simple descent and sine wave horizontal movement
            alien['pos'][1] += alien_speed
            alien['pos'][0] = alien['initial_x'] + math.sin(self.steps * 0.05 + alien['phase']) * 40

            if alien['pos'][1] > self.HEIGHT + 20:
                self.aliens.remove(alien)
                continue

            if self.np_random.random() < alien_fire_rate:
                self._fire_enemy_projectile(alien['pos'])
                # sfx: enemy_shoot
    
    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > -10]
        for p in self.player_projectiles:
            p[1] -= self.PLAYER_PROJ_SPEED

        self.enemy_projectiles = [p for p in self.enemy_projectiles if p[1] < self.HEIGHT + 10]
        for p in self.enemy_projectiles:
            p[1] += self.ENEMY_PROJ_SPEED

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if math.hypot(proj[0] - alien['pos'][0], proj[1] - alien['pos'][1]) < 15:
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self._create_explosion(alien['pos'])
                    self.score += 100
                    reward += 1.0  # Reward for destroying an alien
                    # sfx: alien_explosion
                    break
        
        # Enemy projectiles vs player
        player_hitbox = 10
        for proj in self.enemy_projectiles[:]:
            if math.hypot(proj[0] - self.player_pos[0], proj[1] - self.player_pos[1]) < player_hitbox:
                self.enemy_projectiles.remove(proj)
                self.player_lives -= 1
                self._create_explosion(self.player_pos, count=30)
                reward -= 10.0 # Penalty for getting hit
                # sfx: player_hit
                if self.player_lives <= 0:
                    self.game_over = True
                    self.win_state = False
                    reward -= 100.0 # Penalty for losing
                break
        
        # Check for wave clear
        if not self.aliens and self.wave_clear_timer == 0 and not self.game_over:
            self.wave_clear_timer = self.WAVE_CLEAR_DELAY
            reward += 10.0 # Reward for clearing a wave
            self.score += 1000 * self.current_wave
            
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

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
            "lives": self.player_lives,
            "wave": self.current_wave,
        }

    def _spawn_wave(self):
        rows = 2 + min(self.current_wave, 3)
        cols = 8
        for r in range(rows):
            for c in range(cols):
                x = self.WIDTH * (c + 1) / (cols + 1)
                y = 50 + r * 40
                self.aliens.append({
                    'pos': [x, y],
                    'initial_x': x,
                    'phase': self.np_random.random() * 2 * math.pi
                })

    def _fire_player_projectile(self):
        self.player_projectiles.append(list(self.player_pos))
    
    def _fire_enemy_projectile(self, pos):
        self.enemy_projectiles.append(list(pos))

    def _create_stars(self):
        self.stars = []
        for i in range(200):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'speed': self.np_random.random() * 1.5 + 0.5,
                'size': self.np_random.integers(1, 3)
            })

    def _create_explosion(self, pos, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': random.choice(self.COLOR_EXPLOSION)
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        self._render_stars()
        self._render_projectiles()
        self._render_aliens()
        self._render_player()
        self._render_particles()

    def _render_stars(self):
        for star in self.stars:
            star['pos'][1] = (star['pos'][1] + star['speed']) % self.HEIGHT
            color_val = int(star['speed'] * 50 + 50)
            color = (color_val, color_val, color_val + 20)
            pygame.draw.circle(self.screen, color, star['pos'], star['size'])

    def _render_player(self):
        if self.player_lives > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, 20, self.COLOR_PLAYER_GLOW)
            # Ship body
            points = [(x, y - 12), (x - 10, y + 8), (x + 10, y + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, self.COLOR_ALIEN)
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, (x-12, y-5, 24, 10))

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(p[0]-2), int(p[1]-8), 4, 16))
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 5, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 5, self.COLOR_ENEMY_PROJ)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['life'] / 30.0)
            size = int(alpha * 8)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, p['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))

        # Lives
        for i in range(self.player_lives):
            x, y = self.WIDTH - 20 - i * 25, 20
            points = [(x, y - 6), (x - 5, y + 4), (x + 5, y + 4)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Wave Clear Message
        if self.wave_clear_timer > 0 and not self.game_over:
            msg = "WAVE CLEARED!"
            text = self.font_large.render(msg, True, self.COLOR_WIN_TEXT)
            self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            color = self.COLOR_WIN_TEXT if self.win_state else self.COLOR_LOSE_TEXT
            text = self.font_large.render(msg, True, color)
            self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to "dummy" for headless execution,
    # or remove it to play in a window.
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Game ---")
                obs, info = env.reset()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # In a real game, you might wait for a keypress to reset
            # For this demo, we'll just keep rendering the final state
            # until 'R' is pressed.

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()