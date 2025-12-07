
# Generated: 2025-08-28T06:23:11.873330
# Source Brief: brief_02917.md
# Brief Index: 2917

        
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


class GameEnv(gym.Env):
    """
    A retro-styled, grid-based arcade shooter where the player defends Earth
    from waves of descending aliens. The environment prioritizes visual polish
    and satisfying gameplay feel, with features like particle explosions, a
    parallax starfield, and smooth animations.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    game_description = (
        "Defend Earth from waves of descending aliens in this retro-styled arcade shooter. "
        "Destroy all aliens to advance to the next wave. Survive all 3 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.TOTAL_WAVES = 3

        # Colors
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 100, 255)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_EXPLOSION = [(255, 100, 0), (255, 200, 0), (255, 255, 150)]

        # Gameplay settings
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6  # frames
        self.PLAYER_PROJ_SPEED = 12
        self.ALIEN_PROJ_SPEED = 5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)
        
        # --- Randomness ---
        self.np_random = None

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_rect = None
        self.player_lives = 0
        self.player_fire_timer = 0
        self.wave_number = 0
        self.aliens = []
        self.alien_direction = 1
        self.alien_descent_speed = 0.0
        self.alien_fire_prob = 0.0
        self.projectiles = []
        self.particles = []
        self.stars = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        player_width, player_height = 32, 20
        self.player_rect = pygame.Rect(
            (self.SCREEN_WIDTH - player_width) // 2,
            self.SCREEN_HEIGHT - player_height - 10,
            player_width,
            player_height
        )
        self.player_lives = 3
        self.player_fire_timer = 0

        self.wave_number = 1
        self.projectiles = []
        self.particles = []
        self._generate_stars()
        self._setup_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.metadata["render_fps"])
        reward = 0

        if self.game_over or self.win:
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement, space_held, _ = action
        
        # --- Update Game Logic ---
        reward += self._handle_input(movement, space_held)
        reward += self._update_projectiles()
        wave_reward, wave_cleared = self._update_aliens()
        reward += wave_reward
        
        self._update_particles()
        self._update_stars()

        self.steps += 1
        
        # --- Check Termination Conditions ---
        if self.player_lives <= 0:
            self.game_over = True
            reward -= 100
        
        if wave_cleared and self.wave_number > self.TOTAL_WAVES:
            self.win = True
            reward += 100

        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        if terminated and not (self.game_over or self.win):
            self.game_over = True # Lost by timeout
            reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _setup_wave(self):
        self.aliens.clear()
        self.alien_direction = 1
        
        # Difficulty scaling
        self.alien_descent_speed = 0.5 + 0.2 * (self.wave_number - 1)
        self.alien_fire_prob = 0.005 + 0.002 * (self.wave_number - 1) # per alien per frame

        rows, cols = 4, 10
        alien_w, alien_h = 30, 20
        start_x = (self.SCREEN_WIDTH - cols * (alien_w + 10) + 10) // 2
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (alien_w + 10)
                y = start_y + r * (alien_h + 10)
                self.aliens.append({'rect': pygame.Rect(x, y, alien_w, alien_h), 'alive': True})
    
    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3:  # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_rect.x += self.PLAYER_SPEED
        self.player_rect.clamp_ip(self.screen.get_rect())

        # Firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            proj_rect = pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top, 4, 10)
            self.projectiles.append({'rect': proj_rect, 'type': 'player'})
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
        
        # Continuous movement reward
        if self.aliens:
            alive_aliens_x = [a['rect'].centerx for a in self.aliens if a['alive']]
            if alive_aliens_x:
                alien_center_x = sum(alive_aliens_x) / len(alive_aliens_x)
                
                is_moving_towards = (movement == 3 and self.player_rect.centerx > alien_center_x) or \
                                    (movement == 4 and self.player_rect.centerx < alien_center_x)
                is_moving_away = (movement == 3 and self.player_rect.centerx < alien_center_x) or \
                                 (movement == 4 and self.player_rect.centerx > alien_center_x)

                if is_moving_towards:
                    return 0.1
                elif is_moving_away:
                    return -0.02
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['type'] == 'player':
                proj['rect'].y -= self.PLAYER_PROJ_SPEED
                if proj['rect'].bottom < 0:
                    self.projectiles.remove(proj)
                    continue
                for alien in self.aliens:
                    if alien['alive'] and alien['rect'].colliderect(proj['rect']):
                        # sfx: alien_hit.wav
                        alien['alive'] = False
                        self.projectiles.remove(proj)
                        self._add_particles(alien['rect'].center, 20)
                        self.score += 10
                        reward += 1
                        break
            
            elif proj['type'] == 'alien':
                proj['rect'].y += self.ALIEN_PROJ_SPEED
                if proj['rect'].top > self.SCREEN_HEIGHT:
                    self.projectiles.remove(proj)
                    continue
                if self.player_rect.colliderect(proj['rect']):
                    # sfx: player_hit.wav
                    self.projectiles.remove(proj)
                    self.player_lives -= 1
                    reward -= 1
                    self._add_particles(self.player_rect.center, 30, life=40)
                    if self.player_lives <= 0:
                        self.game_over = True
        return reward

    def _update_aliens(self):
        if not any(a['alive'] for a in self.aliens):
            reward = 10
            self.wave_number += 1
            if self.wave_number <= self.TOTAL_WAVES:
                self._setup_wave()
            return reward, True # Wave cleared

        move_down = False
        for alien in self.aliens:
            if alien['alive']:
                alien['rect'].x += self.alien_direction * (self.alien_descent_speed * 0.5)
                if not (0 < alien['rect'].centerx < self.SCREEN_WIDTH):
                    move_down = True
        
        if move_down:
            self.alien_direction *= -1
            for a in self.aliens:
                a['rect'].y += 15

        for alien in self.aliens:
            if alien['alive']:
                alien['rect'].y += self.alien_descent_speed / self.metadata['render_fps']
                if alien['rect'].bottom > self.player_rect.top:
                    self.game_over = True # Aliens reached player
                
                if self.np_random.random() < self.alien_fire_prob:
                    # sfx: alien_shoot.wav
                    proj_rect = pygame.Rect(alien['rect'].centerx - 2, alien['rect'].bottom, 4, 10)
                    self.projectiles.append({'rect': proj_rect, 'type': 'alien'})
        
        return 0, False

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            speed = self.np_random.random() * 1.5 + 0.5
            size = int(speed)
            self.stars.append({'pos': [x, y], 'speed': speed, 'size': size})

    def _update_stars(self):
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'][0] = self.np_random.integers(0, self.SCREEN_WIDTH)
                star['pos'][1] = 0

    def _add_particles(self, pos, count, life=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': velocity,
                'life': life + self.np_random.integers(-5, 5),
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.rect(self.screen, (255, 255, 255), (int(star['pos'][0]), int(star['pos'][1]), star['size'], star['size']))

        # Aliens
        for alien in self.aliens:
            if alien['alive']:
                pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'])
                # Simple eye details
                eye_y = alien['rect'].centery - 3
                pygame.draw.rect(self.screen, self.COLOR_BG, (alien['rect'].left + 5, eye_y, 5, 5))
                pygame.draw.rect(self.screen, self.COLOR_BG, (alien['rect'].right - 10, eye_y, 5, 5))

        # Projectiles
        for proj in self.projectiles:
            color = self.COLOR_PLAYER_PROJ if proj['type'] == 'player' else self.COLOR_ALIEN_PROJ
            pygame.draw.rect(self.screen, color, proj['rect'])

        # Player
        if self.player_lives > 0:
            # Glow effect
            glow_surf = pygame.Surface((self.player_rect.width * 2, self.player_rect.height * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect())
            self.screen.blit(glow_surf, (self.player_rect.centerx - glow_surf.get_width()//2, self.player_rect.centery - glow_surf.get_height()//2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Ship body
            points = [
                self.player_rect.midtop,
                self.player_rect.bottomleft,
                (self.player_rect.centerx, self.player_rect.centery + 5),
                self.player_rect.bottomright
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life'] / 4), (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.player_lives):
            heart_pos = (self.SCREEN_WIDTH - 80 + i * 25, 17)
            points = [
                (heart_pos[0], heart_pos[1] + 2),
                (heart_pos[0] - 8, heart_pos[1] - 5),
                (heart_pos[0] - 4, heart_pos[1] - 9),
                (heart_pos[0], heart_pos[1] - 5),
                (heart_pos[0] + 4, heart_pos[1] - 9),
                (heart_pos[0] + 8, heart_pos[1] - 5),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Wave
        wave_text = self.font_ui.render(f"WAVE {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH // 2 - wave_text.get_width() // 2, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_ALIEN
        elif self.win:
            msg = "YOU WIN!"
            color = self.COLOR_PLAYER
        else:
            msg = None

        if msg:
            title_text = self.font_title.render(msg, True, color)
            pos = (self.SCREEN_WIDTH // 2 - title_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - title_text.get_height() // 2)
            self.screen.blit(title_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.wave_number,
        }
        
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
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Alien Defender")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running and not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Render to Display ---
        # The observation is already a rendered frame
        # We just need to convert it back to a surface and display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)
            running = False

    env.close()