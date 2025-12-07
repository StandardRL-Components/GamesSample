
# Generated: 2025-08-28T04:52:56.609523
# Source Brief: brief_02459.md
# Brief Index: 2459

        
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
        "Controls: ↑↓←→ to move. Hold shift to drift. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending aliens for 2 minutes in this retro top-down shooter."
    )

    # Frames auto-advance at a fixed rate.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_GAME_TIME = 120  # seconds

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
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_BULLET = (200, 255, 255)
        self.COLOR_ALIEN_BULLET = (255, 200, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.EXPLOSION_COLORS = [(255, 255, 0), (255, 128, 0), (255, 50, 50)]

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.particles = []
        self.stars = []
        self.last_shot_time = 0
        self.next_wave_time = 0
        self.wave_size = 0
        self.difficulty_timer = 0
        self.alien_projectile_speed_bonus = 0.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.MAX_GAME_TIME
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_health = 100
        
        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.particles = []
        
        # Cooldowns and timers
        self.last_shot_time = 0
        self.next_wave_time = 0  # Spawn first wave immediately
        self.difficulty_timer = 0
        
        # Difficulty progression
        self.wave_size = 2
        self.alien_projectile_speed_bonus = 0.0

        # Background
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.choice([1, 2, 3]))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            delta_time = 1 / self.FPS
            self.game_timer -= delta_time

            # --- Base survival reward ---
            reward += 0.01 # Smaller reward to make other actions more significant

            # --- Handle Input and Update Player ---
            self._handle_input(action)

            # --- Update Game Objects ---
            self._update_bullets()
            self._update_aliens()
            self._update_alien_bullets()
            self._update_particles()
            self._update_background()
            
            # --- Handle Collisions ---
            collision_reward = self._handle_collisions()
            reward += collision_reward

            # --- Spawn and Difficulty ---
            self._spawn_waves()
            self._update_difficulty()

            # --- Check Termination ---
            if self.player_health <= 0 or self.game_timer <= 0:
                self.game_over = True
                terminated = True
                if self.player_health > 0: # Survived the timer
                    reward += 100
                else: # Died
                    reward = -100 # Overwrite all other rewards for a strong failure signal
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        player_speed = 6
        drift_multiplier = 1.5 if shift_held else 1.0

        if movement == 1: # Up
            self.player_pos[1] -= player_speed
        if movement == 2: # Down
            self.player_pos[1] += player_speed
        if movement == 3: # Left
            self.player_pos[0] -= player_speed * drift_multiplier
        if movement == 4: # Right
            self.player_pos[0] += player_speed * drift_multiplier

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT - 20)

        # Handle shooting
        fire_cooldown = 0.2  # seconds
        if space_held and (self.steps / self.FPS > self.last_shot_time + fire_cooldown):
            # SFX: Player shoot
            self.last_shot_time = self.steps / self.FPS
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_bullets.append(bullet_pos)

    def _update_bullets(self):
        bullet_speed = 12
        self.player_bullets = [
            [b[0], b[1] - bullet_speed] for b in self.player_bullets if b[1] > 0
        ]

    def _update_aliens(self):
        for alien in self.aliens:
            # Sinusoidal movement
            alien['pos'][1] += alien['speed']
            alien['pos'][0] = alien['base_x'] + math.sin(self.steps * alien['freq']) * alien['amp']
            
            # Alien Firing
            if self.np_random.random() < 0.005: # Firing probability per frame
                # SFX: Alien shoot
                bullet_pos = [alien['pos'][0], alien['pos'][1] + 15]
                self.alien_bullets.append(bullet_pos)

        # Remove aliens that are off-screen
        self.aliens = [a for a in self.aliens if a['pos'][1] < self.HEIGHT]

    def _update_alien_bullets(self):
        base_speed = 4
        speed = base_speed + self.alien_projectile_speed_bonus
        self.alien_bullets = [
            [b[0], b[1] + speed] for b in self.alien_bullets if b[1] < self.HEIGHT
        ]

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_background(self):
        self.stars = [
            (s[0], (s[1] + s[2]) % self.HEIGHT, s[2]) for s in self.stars
        ]

    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs Aliens
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if math.hypot(bullet[0] - alien['pos'][0], bullet[1] - alien['pos'][1]) < 15:
                    # SFX: Explosion
                    self._create_explosion(alien['pos'], 30)
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    self.score += 1
                    reward += 1.0 # Reward for destroying an alien
                    break

        # Alien bullets vs Player
        player_hitbox_radius = 10
        for bullet in self.alien_bullets[:]:
            if math.hypot(bullet[0] - self.player_pos[0], bullet[1] - self.player_pos[1]) < player_hitbox_radius:
                # SFX: Player hit
                self.alien_bullets.remove(bullet)
                self.player_health -= 10
                self._create_explosion(self.player_pos, 10, is_hit=True)
                # No negative reward here to avoid penalizing risk-taking. Failure is handled by terminal reward.
                break
        
        return reward

    def _spawn_waves(self):
        wave_interval = 20 * self.FPS # 20 seconds
        self.next_wave_time -= 1
        if self.next_wave_time <= 0:
            self.next_wave_time = wave_interval
            for _ in range(self.wave_size):
                alien = {
                    'pos': [self.np_random.integers(50, self.WIDTH - 50), self.np_random.integers(-80, -20)],
                    'base_x': self.np_random.integers(50, self.WIDTH - 50),
                    'speed': self.np_random.uniform(0.5, 1.5),
                    'freq': self.np_random.uniform(0.01, 0.05),
                    'amp': self.np_random.uniform(20, 100),
                }
                self.aliens.append(alien)
            self.wave_size += 1

    def _update_difficulty(self):
        difficulty_interval = 30 * self.FPS # 30 seconds
        self.difficulty_timer += 1
        if self.difficulty_timer >= difficulty_interval:
            self.difficulty_timer = 0
            self.alien_projectile_speed_bonus += 0.5 # Was 0.05, increased for more impact

    def _create_explosion(self, pos, num_particles, is_hit=False):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            color = self.COLOR_PLAYER if is_hit else self.np_random.choice(self.EXPLOSION_COLORS)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            color_val = 50 + (speed * 30)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(x), int(y)), speed-1 if speed > 1 else 1)

        # Player Bullets
        for b in self.player_bullets:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, (int(b[0]), int(b[1])), (int(b[0]), int(b[1] - 10)), 3)

        # Alien Bullets
        for b in self.alien_bullets:
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_BULLET, (int(b[0]), int(b[1])), 4)
            pygame.gfxdraw.aacircle(self.screen, int(b[0]), int(b[1]), 4, self.COLOR_ALIEN_BULLET)


        # Aliens
        for a in self.aliens:
            pos = (int(a['pos'][0]), int(a['pos'][1]))
            # Body
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, (pos[0] - 10, pos[1] - 10, 20, 20))
            # Eyes
            pygame.draw.rect(self.screen, self.COLOR_BG, (pos[0] - 6, pos[1] - 6, 4, 4))
            pygame.draw.rect(self.screen, self.COLOR_BG, (pos[0] + 2, pos[1] - 6, 4, 4))
            
        # Player
        p_x, p_y = int(self.player_pos[0]), int(self.player_pos[1])
        player_points = [(p_x, p_y - 15), (p_x - 12, p_y + 10), (p_x + 12, p_y + 10)]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, player_points)
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)

        # Engine trail
        if self.np_random.random() < 0.8:
            vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 4)]
            life = self.np_random.integers(5, 10)
            self.particles.append({'pos': [p_x, p_y + 10], 'vel': vel, 'life': life, 'color': self.COLOR_PLAYER_BULLET})

        # Particles
        for p in self.particles:
            size = max(0, int(p['life'] / 5))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {max(0, int(self.game_timer))}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Health Bar
        health_bar_width = 100
        health_pct = max(0, self.player_health / 100.0)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(health_bar_width * health_pct), 20))

        # Game Over Message
        if self.game_over:
            if self.player_health > 0:
                msg = "YOU WIN"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ALIEN
            
            over_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(over_text, (self.WIDTH // 2 - over_text.get_width() // 2, self.HEIGHT // 2 - over_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "timer": self.game_timer,
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display for human interaction
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Galactic Survival")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Rendering for human display ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            # Keep displaying the final screen until R is pressed
            pass

        clock.tick(env.FPS)
        
    env.close()