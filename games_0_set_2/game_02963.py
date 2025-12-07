
# Generated: 2025-08-28T06:32:26.858997
# Source Brief: brief_02963.md
# Brief Index: 2963

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move. Hold space to fire your weapon. Defend Earth from the alien horde!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from a descending alien horde in this fast-paced, top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.TOTAL_WAVES = 5

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_BULLET = (255, 255, 100)
        self.COLOR_ALIEN_BULLET = (255, 50, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.ALIEN_COLORS = [
            (255, 50, 50), (255, 150, 50), (50, 150, 255), 
            (200, 50, 255), (255, 255, 50)
        ]

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game specific attributes, initialized in reset
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.player_pos = None
        self.player_lives = None
        self.player_bullets = None
        self.alien_bullets = None
        self.aliens = None
        self.particles = None
        self.stars = None
        self.current_wave = None
        self.alien_projectile_speed = None
        self.last_shot_time = None
        self.invincibility_timer = None
        self.alien_direction = None
        self.alien_y_descent_timer = None

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.particles = []
        
        self.current_wave = 1
        self.alien_projectile_speed = 2.0
        self.last_shot_time = 0
        self.invincibility_timer = 120 # Start with brief invincibility
        
        self.alien_direction = 1
        self.alien_y_descent_timer = 0
        
        self._spawn_stars()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty for time to encourage fast play
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._handle_collisions()
            reward += self._check_wave_completion()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.win:
             reward -= 100 # Adjusted from brief to fit scale
        elif terminated and self.win:
             reward += 100 # Adjusted from brief to fit scale
        
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Player Movement
        player_speed = 8
        if movement == 3: # Left
            self.player_pos[0] -= player_speed
        elif movement == 4: # Right
            self.player_pos[0] += player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        # Player Firing
        fire_rate_cooldown = 6 # frames
        if space_held and (self.steps - self.last_shot_time) > fire_rate_cooldown:
            # SFX: Player shoot
            self.last_shot_time = self.steps
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_bullets.append({'pos': bullet_pos, 'vel': [0, -12]})
            # Muzzle flash
            for _ in range(10):
                self.particles.append(self._create_particle(
                    pos=[bullet_pos[0], bullet_pos[1] + 5],
                    base_color=(255, 255, 100),
                    life=random.randint(5, 10),
                    size=random.uniform(1, 3),
                    velocity_scale=2
                ))

    def _update_game_state(self):
        # Update invincibility
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            
        # Update player bullets
        for bullet in self.player_bullets[:]:
            bullet['pos'][0] += bullet['vel'][0]
            bullet['pos'][1] += bullet['vel'][1]
            if bullet['pos'][1] < 0:
                self.player_bullets.remove(bullet)
        
        # Update alien bullets
        for bullet in self.alien_bullets[:]:
            bullet['pos'][0] += bullet['vel'][0]
            bullet['pos'][1] += bullet['vel'][1]
            if bullet['pos'][1] > self.HEIGHT:
                self.alien_bullets.remove(bullet)

        # Update aliens
        self._update_aliens()

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_aliens(self):
        move_sideways = False
        drop_down = False
        
        alien_speed = 1.0 + self.current_wave * 0.2
        
        for alien in self.aliens:
            if (alien['pos'][0] > self.WIDTH - 20 and self.alien_direction > 0) or \
               (alien['pos'][0] < 20 and self.alien_direction < 0):
                move_sideways = True
                break
        
        if move_sideways:
            self.alien_direction *= -1
            drop_down = True
            
        for alien in self.aliens:
            alien['pos'][0] += self.alien_direction * alien_speed
            if drop_down:
                alien['pos'][1] += 20
            
            # Alien firing
            fire_chance = 0.001 + self.current_wave * 0.0005
            if self.np_random.random() < fire_chance:
                # SFX: Alien shoot
                bullet_pos = [alien['pos'][0], alien['pos'][1] + 10]
                self.alien_bullets.append({'pos': bullet_pos, 'vel': [0, self.alien_projectile_speed]})

    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs Aliens
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if math.hypot(bullet['pos'][0] - alien['pos'][0], bullet['pos'][1] - alien['pos'][1]) < 15:
                    # SFX: Explosion
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    self.score += 100
                    reward += 1.0
                    self._create_explosion(alien['pos'], alien['color'])
                    break
        
        # Alien bullets vs Player
        if self.invincibility_timer <= 0:
            player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)
            for bullet in self.alien_bullets[:]:
                if player_rect.collidepoint(bullet['pos'][0], bullet['pos'][1]):
                    self.alien_bullets.remove(bullet)
                    reward += self._player_hit()
                    break
        
        # Aliens vs Player
        if self.invincibility_timer <= 0:
             for alien in self.aliens:
                if math.hypot(self.player_pos[0] - alien['pos'][0], self.player_pos[1] - alien['pos'][1]) < 25:
                    reward += self._player_hit()
                    break

        # Aliens vs bottom of screen
        for alien in self.aliens:
            if alien['pos'][1] > self.HEIGHT - 50:
                reward += self._player_hit() # Reaching bottom is same as a hit
                self.aliens.remove(alien) # Remove the one that reached
                self._create_explosion(alien['pos'], alien['color'])


        return reward

    def _player_hit(self):
        # SFX: Player hit/explosion
        self.player_lives -= 1
        self.invincibility_timer = 120 # 4 seconds of invincibility
        self._create_explosion(self.player_pos, self.COLOR_PLAYER)
        if self.player_lives <= 0:
            self.game_over = True
            return -10.0 # Large penalty for losing the game
        return -5.0 # Penalty for losing a life

    def _check_wave_completion(self):
        if not self.aliens and not self.game_over:
            # SFX: Wave clear
            self.current_wave += 1
            if self.current_wave > self.TOTAL_WAVES:
                self.win = True
                self.game_over = True
                return 50.0 # Large reward for winning
            else:
                self.alien_projectile_speed += 0.5
                self._spawn_wave()
                self.invincibility_timer = 90 # Invincibility between waves
                return 10.0 # Reward for clearing a wave
        return 0

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for star in self.stars:
            color_val = star['brightness']
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(star['x']), int(star['y'])), star['size'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))
            
        # Draw bullets
        for bullet in self.player_bullets:
            pos = (int(bullet['pos'][0]), int(bullet['pos'][1]))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, pos, (pos[0], pos[1] - 10), 3)
        for bullet in self.alien_bullets:
            pos = (int(bullet['pos'][0]), int(bullet['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ALIEN_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ALIEN_BULLET)

        # Draw aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            color = alien['color']
            points = [(x, y - 8), (x - 8, y + 8), (x + 8, y + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw player
        is_visible = self.invincibility_timer <= 0 or (self.steps % 10 < 5)
        if is_visible and self.player_lives > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            points = [(x, y - 12), (x - 12, y + 10), (x + 12, y + 10)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            # Engine glow
            if self.invincibility_timer > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, 20, (*self.COLOR_PLAYER, 50))
                pygame.gfxdraw.aacircle(self.screen, x, y, 20, (*self.COLOR_PLAYER, 100))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            x, y = 20 + i * 30, 25
            points = [(x, y - 8), (x - 8, y + 8), (x + 8, y + 8)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.ALIEN_COLORS[0]
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
        }

    def _spawn_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'x': self.np_random.integers(0, self.WIDTH),
                'y': self.np_random.integers(0, self.HEIGHT),
                'size': self.np_random.integers(1, 3),
                'brightness': self.np_random.integers(50, 150)
            })

    def _spawn_wave(self):
        rows = 2 + min(self.current_wave, 3)
        cols = 8
        x_spacing = 60
        y_spacing = 40
        start_x = (self.WIDTH - (cols - 1) * x_spacing) // 2
        start_y = 60
        for r in range(rows):
            for c in range(cols):
                pos = [start_x + c * x_spacing, start_y + r * y_spacing]
                color = self.ALIEN_COLORS[(r + self.current_wave - 1) % len(self.ALIEN_COLORS)]
                self.aliens.append({'pos': pos, 'color': color})

    def _create_particle(self, pos, base_color, life, size, velocity_scale):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5) * velocity_scale
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        return {
            'pos': list(pos), 'vel': vel, 'size': size,
            'color': base_color, 'life': life, 'max_life': life
        }

    def _create_explosion(self, pos, base_color):
        for _ in range(30):
            self.particles.append(self._create_particle(
                pos=pos, base_color=base_color,
                life=random.randint(15, 30),
                size=random.uniform(1, 4),
                velocity_scale=3
            ))

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame headless
    env = GameEnv()
    
    # To run headlessly and check for errors
    print("Running a short headless test...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Headless test finished. Final info: {info}, Total Reward: {total_reward}")
    
    # To run with a window and play
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        
        render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Alien Horde Defender")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            # Shift is not used in this game
            # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            #     shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Score: {info['score']}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
    except pygame.error as e:
        print("\nCould not create display. Pygame might not be configured for windowed mode.")
        print("This is normal in some environments (e.g., cloud instances).")
        print("The headless test completed successfully.")

    pygame.quit()