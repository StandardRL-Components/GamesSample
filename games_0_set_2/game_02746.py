
# Generated: 2025-08-27T21:18:52.533404
# Source Brief: brief_02746.md
# Brief Index: 2746

        
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
    A top-down arcade shooter Gymnasium environment. The player controls a ship
    at the bottom of the screen, moving horizontally and firing upwards to
    destroy waves of descending aliens. The game prioritizes visual polish
    and a satisfying "game feel" with particle effects, smooth motion, and
    responsive controls.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. Hold space to fire your weapon."
    )

    game_description = (
        "A fast-paced, retro-style arcade shooter. Destroy waves of descending "
        "aliens, dodge their fire, and aim for a high score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 40)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_ALIEN_GLOW = (255, 50, 50, 40)
        self.COLOR_PLAYER_BULLET = (200, 255, 255)
        self.COLOR_ALIEN_BULLET = (255, 100, 200)
        self.COLOR_PARTICLE_YELLOW = (255, 255, 0)
        self.COLOR_PARTICLE_ORANGE = (255, 150, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 80, 80)

        # Game settings
        self.PLAYER_SPEED = 10
        self.PLAYER_FIRE_COOLDOWN = 5  # frames
        self.PLAYER_BULLET_SPEED = 15
        self.PLAYER_HITBOX_RADIUS = 12
        self.ALIEN_BULLET_SPEED = 5
        self.ALIEN_HITBOX_RADIUS = 12

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 36, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_cooldown_timer = 0
        self.player_invincibility_timer = 0
        self.wave = 0
        self.wave_transition_timer = 0
        
        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.particles = []
        self.stars = []
        
        self.alien_move_dir = 1
        self.alien_speed = 0
        self.alien_fire_prob = 0
        self.alien_descent_speed = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        self.player_invincibility_timer = 0
        self.wave = 0 # Will be incremented to 1 in _spawn_wave
        self.wave_transition_timer = self.FPS * 2 # Show "Wave 1"

        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.particles = []

        self.alien_move_dir = 1
        
        # Create a static starfield for visual appeal
        if not self.stars:
            for _ in range(200):
                self.stars.append({
                    'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                    'speed': self.np_random.uniform(0.2, 1.0),
                    'size': self.np_random.integers(1, 3),
                    'color': self.np_random.integers(50, 150)
                })

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Small penalty per step to encourage efficiency

        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
        else:
            reward += self._update_game_logic(action)

        self._update_visual_effects()
        
        self.steps += 1
        terminated = self.player_lives <= 0 or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_lives <= 0:
                reward -= 100  # Large penalty for losing

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_logic(self, action):
        frame_reward = 0
        
        self._handle_input(action)
        
        # Update timers
        if self.player_fire_cooldown_timer > 0: self.player_fire_cooldown_timer -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1
        
        # Update entities and check for collisions
        frame_reward += self._update_player_bullets()
        frame_reward += self._update_aliens()
        frame_reward += self._update_alien_bullets()
        
        # Check for wave clear
        if not self.aliens and not self.game_over:
            frame_reward += 5
            self.score += 100 # Wave clear bonus
            self.wave_transition_timer = self.FPS * 2
            self._spawn_wave()

        return frame_reward
    
    def _update_visual_effects(self):
        # Update starfield for parallax effect
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][0] = self.np_random.integers(0, self.WIDTH)
                star['pos'][1] = 0

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)

        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: Player shoot
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 20]
            self.player_bullets.append(bullet_pos)
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player_bullets(self):
        reward = 0
        for bullet in self.player_bullets[:]:
            bullet[1] -= self.PLAYER_BULLET_SPEED
            if bullet[1] < 0:
                self.player_bullets.remove(bullet)
                continue
            
            for alien in self.aliens[:]:
                dist = math.hypot(bullet[0] - alien['pos'][0], bullet[1] - alien['pos'][1])
                if dist < self.ALIEN_HITBOX_RADIUS + 5: # 5 for bullet size
                    # SFX: Alien explosion
                    self._create_explosion(alien['pos'], self.COLOR_ALIEN)
                    self.aliens.remove(alien)
                    self.player_bullets.remove(bullet)
                    self.score += 10
                    reward += 1
                    break
        return reward

    def _update_aliens(self):
        move_down = False
        for alien in self.aliens:
            if (alien['pos'][0] > self.WIDTH - 20 and self.alien_move_dir > 0) or \
               (alien['pos'][0] < 20 and self.alien_move_dir < 0):
                move_down = True
                break
        
        if move_down:
            self.alien_move_dir *= -1
            for a in self.aliens:
                a['pos'][1] += 15
        
        for alien in self.aliens:
            alien['pos'][0] += self.alien_move_dir * self.alien_speed
            alien['pos'][1] += self.alien_descent_speed

            # Check if aliens reached player's vertical line
            if alien['pos'][1] > self.HEIGHT - 20:
                self.player_lives = 0 # Instant loss

            if self.np_random.random() < self.alien_fire_prob:
                # SFX: Alien shoot
                bullet_pos = [alien['pos'][0], alien['pos'][1] + 20]
                self.alien_bullets.append({'pos': bullet_pos, 'reward_pending': True})
        return 0

    def _update_alien_bullets(self):
        reward = 0
        for bullet in self.alien_bullets[:]:
            bullet['pos'][1] += self.ALIEN_BULLET_SPEED
            
            # Remove off-screen bullets
            if bullet['pos'][1] > self.HEIGHT:
                self.alien_bullets.remove(bullet)
                continue
            
            # Check for player collision
            dist = math.hypot(bullet['pos'][0] - self.player_pos[0], bullet['pos'][1] - self.player_pos[1])
            if dist < self.PLAYER_HITBOX_RADIUS and self.player_invincibility_timer == 0:
                # SFX: Player explosion
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                self.player_lives -= 1
                self.player_invincibility_timer = self.FPS * 2 # 2 seconds of invincibility
                self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40] # Reset position
                self.alien_bullets.remove(bullet)
                continue

            # Check for dodge reward
            if bullet['reward_pending'] and bullet['pos'][1] > self.player_pos[1]:
                reward += 0.1
                bullet['reward_pending'] = False
        return reward

    def _spawn_wave(self):
        self.wave += 1
        self.aliens.clear()
        self.alien_bullets.clear()
        
        self.alien_speed = 1.0 + (self.wave - 1) * 0.05
        self.alien_fire_prob = min(0.1, 0.005 + (self.wave - 1) * 0.001)
        self.alien_descent_speed = 0.05 + (self.wave - 1) * 0.01

        rows = min(5, 2 + self.wave // 2)
        cols = min(10, 6 + self.wave // 3)
        
        grid_width = cols * 50
        start_x = (self.WIDTH - grid_width) / 2 + 25
        
        for r in range(rows):
            for c in range(cols):
                pos = [start_x + c * 50, 50 + r * 40]
                self.aliens.append({'pos': pos})

    def _create_explosion(self, pos, base_color):
        num_particles = 25
        r, g, b = base_color
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(15, 30)
            color_choice = self.np_random.choice([self.COLOR_PARTICLE_YELLOW, self.COLOR_PARTICLE_ORANGE, (r,g,b)])
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color_choice
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for star in self.stars:
            c = star['color']
            pygame.draw.rect(self.screen, (c,c,c), (int(star['pos'][0]), int(star['pos'][1]), star['size'], star['size']))

    def _render_game(self):
        # Draw player bullets
        for b in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, (int(b[0] - 2), int(b[1] - 10), 4, 20), border_radius=2)
            
        # Draw alien bullets
        for b in self.alien_bullets:
            pygame.gfxdraw.filled_circle(self.screen, int(b['pos'][0]), int(b['pos'][1]), 5, self.COLOR_ALIEN_BULLET)
            pygame.gfxdraw.aacircle(self.screen, int(b['pos'][0]), int(b['pos'][1]), 5, self.COLOR_ALIEN_BULLET)

        # Draw aliens
        for a in self.aliens:
            x, y = int(a['pos'][0]), int(a['pos'][1])
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ALIEN_HITBOX_RADIUS + 4, self.COLOR_ALIEN_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ALIEN_HITBOX_RADIUS, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ALIEN_HITBOX_RADIUS, self.COLOR_ALIEN)

        # Draw player
        if self.player_lives > 0:
            is_invincible = self.player_invincibility_timer > 0
            if not (is_invincible and (self.player_invincibility_timer // 3) % 2 == 0):
                x, y = int(self.player_pos[0]), int(self.player_pos[1])
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.PLAYER_HITBOX_RADIUS + 8, self.COLOR_PLAYER_GLOW)
                # Ship body
                points = [(x, y - 15), (x - 12, y + 10), (x + 12, y + 10)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life'] * 0.3), color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            x, y = 20 + i * 30, 25
            points = [(x, y-10), (x-10, y), (x, y+10), (x+10, y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

        # Wave transition text
        if self.wave_transition_timer > 0:
            alpha = min(255, int(255 * (self.wave_transition_timer / (self.FPS * 0.5))))
            wave_text = self.font_wave.render(f"WAVE {self.wave}", True, self.COLOR_UI_TEXT)
            wave_text.set_alpha(alpha)
            text_rect = wave_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(wave_text, text_rect)

        # Game Over text
        if self.game_over:
            game_over_text = self.font_game_over.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.wave,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate pygame window for human play
    pygame.display.set_caption("Arcade Shooter")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        mov = 0 # No-op
        if keys[pygame.K_LEFT]:
            mov = 3
        elif keys[pygame.K_RIGHT]:
            mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    env.close()