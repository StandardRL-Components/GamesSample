
# Generated: 2025-08-27T17:48:27.261937
# Source Brief: brief_01644.md
# Brief Index: 1644

        
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
    """
    A retro top-down arcade shooter Gymnasium environment.

    The player controls a ship at the bottom of the screen and must survive
    for 180 seconds against waves of descending alien invaders. The player
    can move, shoot, and activate a temporary shield. Difficulty increases
    over time with more aliens and faster projectiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to fire. Press Shift to activate your shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending alien invaders in this retro top-down shooter. "
        "Dodge enemy fire, destroy aliens, and use your shield wisely to last 3 minutes."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 180
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS + 10 # A little buffer

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 40)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_BULLET = (255, 255, 0)
        self.COLOR_ALIEN_BULLET = (200, 0, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (255, 0, 0)]
        self.COLOR_SHIELD = (100, 150, 255)
        self.COLOR_SHIELD_GLOW = (100, 150, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_TIMER = (255, 255, 100)
        self.COLOR_UI_COOLDOWN = (100, 100, 100)
        self.COLOR_STAR = (200, 200, 220)

        # Game Parameters
        self.PLAYER_SPEED = 7
        self.PLAYER_FIRE_RATE = 4  # frames per shot
        self.PLAYER_BULLET_SPEED = 12
        self.PLAYER_SIZE = 20
        self.ALIEN_SIZE = 22
        self.ALIEN_BULLET_BASE_SPEED = 4
        self.SHIELD_DURATION_FRAMES = int(3 * self.FPS)
        self.SHIELD_COOLDOWN_FRAMES = int(10 * self.FPS)
        self.INITIAL_ALIEN_SPAWN_INTERVAL_S = 1.0

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 28, bold=True)
        except pygame.error:
            self.font_small = pygame.font.SysFont(None, 22)
            self.font_large = pygame.font.SysFont(None, 32)
        
        # State variables initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed_frames = 0
        self.player_rect = None
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.explosions = []
        self.stars = []
        self.alien_spawn_timer = 0
        self.current_alien_spawn_interval = 0
        self.current_alien_bullet_speed = 0

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed_frames = 0

        self.player_rect = pygame.Rect(self.WIDTH // 2 - self.PLAYER_SIZE // 2, self.HEIGHT - 50, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_lives = 3
        self.player_fire_cooldown = 0

        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.explosions = []

        self.alien_spawn_timer = 0
        self.current_alien_spawn_interval = self.INITIAL_ALIEN_SPAWN_INTERVAL_S * self.FPS
        self.current_alien_bullet_speed = self.ALIEN_BULLET_BASE_SPEED

        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 4))
            for _ in range(150)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        self.time_elapsed_frames += 1
        time_elapsed_seconds = self.time_elapsed_frames / self.FPS

        # --- Reward for surviving this frame ---
        reward = 0.1

        # --- Handle Input & Update Player ---
        self._handle_input(action)

        # --- Update Game State ---
        self._update_shield()
        self._update_bullets()
        self._update_aliens(time_elapsed_seconds)
        self._update_effects()

        # --- Handle Collisions & Calculate Event Rewards ---
        reward += self._handle_collisions()

        # --- Check Termination Conditions ---
        win = time_elapsed_seconds >= self.GAME_DURATION_SECONDS
        lose = self.player_lives <= 0
        timeout = self.steps >= self.MAX_STEPS

        terminated = win or lose or timeout
        if terminated:
            self.game_over = True
            if win and self.player_lives > 0:
                reward += 50  # Win bonus

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_rect.y -= self.PLAYER_SPEED
        if movement == 2: self.player_rect.y += self.PLAYER_SPEED
        if movement == 3: self.player_rect.x -= self.PLAYER_SPEED
        if movement == 4: self.player_rect.x += self.PLAYER_SPEED
        self.player_rect.clamp_ip(self.screen.get_rect())

        # Firing
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if space_held and self.player_fire_cooldown == 0:
            bullet_rect = pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top, 4, 10)
            self.player_bullets.append(bullet_rect)
            self.player_fire_cooldown = self.PLAYER_FIRE_RATE
            # sfx: player_shoot

        # Shield
        if shift_held and not self.shield_active and self.shield_cooldown_timer == 0:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION_FRAMES
            # sfx: shield_activate

    def _update_shield(self):
        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
                self.shield_cooldown_timer = self.SHIELD_COOLDOWN_FRAMES
                # sfx: shield_deactivate
        elif self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1

    def _update_bullets(self):
        self.player_bullets = [b for b in self.player_bullets if b.move(0, -self.PLAYER_BULLET_SPEED).bottom > 0]
        self.alien_bullets = [b for b in self.alien_bullets if b.move(0, self.current_alien_bullet_speed).top < self.HEIGHT]

    def _update_aliens(self, time_elapsed_seconds):
        # Difficulty scaling
        difficulty_tier_spawn = time_elapsed_seconds // 30
        self.current_alien_spawn_interval = max(0.2 * self.FPS, (self.INITIAL_ALIEN_SPAWN_INTERVAL_S - difficulty_tier_spawn * 0.15) * self.FPS)
        
        difficulty_tier_speed = time_elapsed_seconds // 60
        self.current_alien_bullet_speed = self.ALIEN_BULLET_BASE_SPEED + difficulty_tier_speed * 1.5

        # Spawn new aliens
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            self._spawn_alien()
            self.alien_spawn_timer = self.current_alien_spawn_interval

        # Move existing aliens
        for alien in self.aliens[:]:
            alien['pos'][1] += 2 # Base downward speed
            if alien['type'] == 'zigzag':
                alien['pos'][0] += alien['speed']
                if not (0 < alien['pos'][0] < self.WIDTH - self.ALIEN_SIZE):
                    alien['speed'] *= -1
            elif alien['type'] == 'sine':
                alien['phase'] += 0.05
                alien['pos'][0] = alien['centerx'] + math.sin(alien['phase']) * 50
            
            alien['rect'].topleft = alien['pos']

            # Alien firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                bullet_rect = pygame.Rect(alien['rect'].centerx - 3, alien['rect'].bottom, 6, 12)
                self.alien_bullets.append(bullet_rect)
                alien['fire_cooldown'] = self.np_random.integers(60, 120)
                # sfx: alien_shoot

            if alien['rect'].top > self.HEIGHT:
                self.aliens.remove(alien)

    def _spawn_alien(self):
        alien_type = self.np_random.choice(['zigzag', 'sine'])
        start_x = self.np_random.integers(self.ALIEN_SIZE, self.WIDTH - self.ALIEN_SIZE)
        
        alien = {
            'rect': pygame.Rect(start_x, -self.ALIEN_SIZE, self.ALIEN_SIZE, self.ALIEN_SIZE),
            'pos': [float(start_x), float(-self.ALIEN_SIZE)],
            'type': alien_type,
            'fire_cooldown': self.np_random.integers(30, 90)
        }
        if alien_type == 'zigzag':
            alien['speed'] = self.np_random.choice([-2, 2])
        else: # sine
            alien['centerx'] = start_x
            alien['phase'] = self.np_random.uniform(0, 2 * math.pi)
        
        self.aliens.append(alien)

    def _update_effects(self):
        for explosion in self.explosions[:]:
            explosion['timer'] -= 1
            if explosion['timer'] <= 0:
                self.explosions.remove(explosion)

    def _handle_collisions(self):
        reward = 0

        # Player bullets vs Aliens
        for bullet in self.player_bullets[:]:
            hit_alien_index = bullet.collidelist([a['rect'] for a in self.aliens])
            if hit_alien_index != -1:
                self.player_bullets.remove(bullet)
                hit_alien = self.aliens.pop(hit_alien_index)
                self._create_explosion(hit_alien['rect'].center)
                self.score += 100
                reward += 1
                # sfx: alien_explosion
                break

        # Alien bullets vs Player
        for bullet in self.alien_bullets[:]:
            if self.player_rect.colliderect(bullet):
                self.alien_bullets.remove(bullet)
                if self.shield_active:
                    reward += 5
                    # sfx: shield_block
                    # Create a smaller "block" effect
                    self._create_explosion(self.player_rect.center, size=15, duration=10, is_shield=True)
                else:
                    self._player_hit()
                    reward -= 5
                break

        # Aliens vs Player
        for alien in self.aliens[:]:
            if self.player_rect.colliderect(alien['rect']):
                self.aliens.remove(alien)
                self._create_explosion(alien['rect'].center)
                if not self.shield_active:
                    self._player_hit()
                    reward -= 5
                break # Only one collision per frame
        
        return reward

    def _player_hit(self):
        self.player_lives -= 1
        self._create_explosion(self.player_rect.center, size=40, duration=25)
        self.player_rect.centerx = self.WIDTH // 2 # Respawn in center
        # sfx: player_hit
        if self.player_lives <= 0:
            self.game_over = True

    def _create_explosion(self, pos, size=30, duration=20, is_shield=False):
        self.explosions.append({'pos': pos, 'max_size': size, 'timer': duration, 'max_timer': duration, 'is_shield': is_shield})

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i, (x, y, size) in enumerate(self.stars):
            y_new = (y + (size * 0.5) * 1) % self.HEIGHT
            self.stars[i] = (x, y_new, size)
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, int(y_new), size, size))

    def _render_game_elements(self):
        # Aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'])

        # Bullets
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet)
        for bullet in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_BULLET, bullet)

        # Player
        if self.player_lives > 0:
            # Draw glow
            glow_radius = int(self.PLAYER_SIZE * 1.5)
            pygame.gfxdraw.filled_circle(self.screen, self.player_rect.centerx, self.player_rect.centery, glow_radius, self.COLOR_PLAYER_GLOW)
            # Draw ship
            p = self.player_rect
            points = [(p.centerx, p.top), (p.left, p.bottom), (p.right, p.bottom)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            
            # Shield
            if self.shield_active:
                shield_radius = int(self.PLAYER_SIZE * 0.8)
                # Animate pulsing effect
                pulse = abs(math.sin(self.time_elapsed_frames * 0.3))
                current_radius = int(shield_radius + pulse * 5)
                pygame.gfxdraw.filled_circle(self.screen, p.centerx, p.centery, current_radius, self.COLOR_SHIELD_GLOW)
                pygame.gfxdraw.aacircle(self.screen, p.centerx, p.centery, current_radius, self.COLOR_SHIELD)

        # Explosions
        for exp in self.explosions:
            progress = 1 - (exp['timer'] / exp['max_timer'])
            radius = int(exp['max_size'] * progress)
            alpha = int(255 * (1 - progress))
            if exp['is_shield']:
                color = (*self.COLOR_SHIELD, alpha)
            else:
                color_idx = min(2, int(progress * 3))
                color = (*self.COLOR_EXPLOSION[color_idx], alpha)
            
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.time_elapsed_frames / self.FPS))
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = self.font_large.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_UI_TIMER)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))
        
        # Lives
        for i in range(self.player_lives):
            p_size = self.PLAYER_SIZE * 0.75
            p_rect = pygame.Rect(self.WIDTH // 2 - (self.player_lives * (p_size + 5)) // 2 + i * (p_size + 5), self.HEIGHT - p_size - 10, p_size, p_size)
            points = [(p_rect.centerx, p_rect.top), (p_rect.left, p_rect.bottom), (p_rect.right, p_rect.bottom)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Shield Cooldown Indicator
        shield_icon_pos = (20, self.HEIGHT - 30)
        if self.shield_cooldown_timer > 0:
            color = self.COLOR_UI_COOLDOWN
            # Cooldown pie
            progress = self.shield_cooldown_timer / self.SHIELD_COOLDOWN_FRAMES
            angle = int(360 * progress)
            rect = pygame.Rect(shield_icon_pos[0] - 12, shield_icon_pos[1] - 12, 24, 24)
            pygame.draw.arc(self.screen, self.COLOR_SHIELD, rect, math.radians(90), math.radians(90 + angle), 3)
        elif self.shield_active:
            color = self.COLOR_SHIELD
        else:
            color = self.COLOR_UI_TEXT
        
        pygame.gfxdraw.aacircle(self.screen, shield_icon_pos[0], shield_icon_pos[1], 10, color)
        pygame.gfxdraw.filled_circle(self.screen, shield_icon_pos[0], shield_icon_pos[1], 10, (*color, 50))
        shield_text = self.font_small.render("SHIELD", True, color)
        self.screen.blit(shield_text, (shield_icon_pos[0] + 20, shield_icon_pos[1] - 9))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Galactic Defender")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print("\n" + "="*30)
    print(f"GAME: Galactic Defender")
    print(f"INFO: {env.game_description}")
    print(f"CTRL: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] or keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            running = False

        clock.tick(env.FPS)

    env.close()