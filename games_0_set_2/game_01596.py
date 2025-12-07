
# Generated: 2025-08-27T17:38:33.258698
# Source Brief: brief_01596.md
# Brief Index: 1596

        
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
    A retro top-down arcade shooter where the player defends against a descending alien horde.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from a descending alien horde in this retro top-down shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 40)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_BULLET = (255, 255, 100)
        self.COLOR_ALIEN_BULLET = (255, 100, 255)
        self.COLOR_EXPLOSION_1 = (255, 200, 0)
        self.COLOR_EXPLOSION_2 = (255, 100, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Game parameters
        self.PLAYER_SPEED = 8
        self.PLAYER_LIVES = 3
        self.PLAYER_INVINCIBILITY_FRAMES = 90  # 3 seconds at 30fps
        self.PLAYER_BULLET_SPEED = 12
        self.ALIEN_ROWS = 3
        self.ALIEN_COLS = 10
        self.TOTAL_ALIENS = self.ALIEN_ROWS * self.ALIEN_COLS
        self.INITIAL_ALIEN_SPEED = 0.5
        self.INITIAL_ALIEN_FIRE_RATE = 0.002
        self.ALIEN_DROP_AMOUNT = 10

        # --- Gymnasium API Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except pygame.error:
            # Fallback font if default is not found
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_game_over = pygame.font.SysFont("monospace", 48)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.player_lives = None
        self.player_pos = None
        self.player_hit_timer = None
        self.last_space_held = None
        self.aliens = None
        self.alien_direction = None
        self.alien_speed = None
        self.alien_fire_rate = None
        self.aliens_destroyed = None
        self.player_bullets = None
        self.alien_bullets = None
        self.explosions = None
        self.game_over = None
        self.win = None

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_lives = self.PLAYER_LIVES
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float32)
        self.player_hit_timer = 0
        self.last_space_held = False

        self.player_bullets = []
        self.alien_bullets = []
        self.explosions = []

        self.aliens = []
        alien_w, alien_h = 20, 20
        start_x = (self.WIDTH - self.ALIEN_COLS * (alien_w + 20)) / 2
        for r in range(self.ALIEN_ROWS):
            for c in range(self.ALIEN_COLS):
                x = start_x + c * (alien_w + 20)
                y = 50 + r * (alien_h + 15)
                self.aliens.append({'pos': np.array([x, y], dtype=np.float32), 'alive': True})

        self.alien_direction = 1.0
        self.aliens_destroyed = 0
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Continuous reward for surviving
        self.steps += 1

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)

        if space_held and not self.last_space_held:
            # SFX: Player shoot
            self.player_bullets.append(self.player_pos.copy())
        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_player()
        self._update_bullets()
        self._update_aliens()
        self._update_explosions()

        # --- Collision Detection ---
        bullet_reward, hit_penalty = self._handle_collisions()
        reward += bullet_reward + hit_penalty

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_lives <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            self.win = False
        elif self.aliens_destroyed == self.TOTAL_ALIENS:
            reward = 100.0
            terminated = True
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = False

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_player(self):
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1
            
    def _update_bullets(self):
        self.player_bullets = [b + np.array([0, -self.PLAYER_BULLET_SPEED]) for b in self.player_bullets]
        self.player_bullets = [b for b in self.player_bullets if b[1] > 0]
        self.alien_bullets = [b + np.array([0, self.PLAYER_BULLET_SPEED / 2]) for b in self.alien_bullets]
        self.alien_bullets = [b for b in self.alien_bullets if b[1] < self.HEIGHT]

    def _update_aliens(self):
        move_down = False
        for alien in self.aliens:
            if alien['alive']:
                alien['pos'][0] += self.alien_speed * self.alien_direction
                if alien['pos'][0] <= 15 or alien['pos'][0] >= self.WIDTH - 15:
                    move_down = True
        
        if move_down:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien['pos'][1] += self.ALIEN_DROP_AMOUNT

        living_aliens = [a for a in self.aliens if a['alive']]
        if living_aliens and self.np_random.random() < self.alien_fire_rate * len(living_aliens):
            shooter = self.np_random.choice(living_aliens)
            # SFX: Alien shoot
            self.alien_bullets.append(shooter['pos'].copy())
            
        for alien in self.aliens:
            if alien['alive'] and alien['pos'][1] > self.HEIGHT - 60:
                self.player_lives = 0 # Instant loss if aliens reach player
                break

    def _update_explosions(self):
        for exp in self.explosions:
            exp['timer'] -= 1
        self.explosions = [exp for exp in self.explosions if exp['timer'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs aliens
        bullets_to_remove = set()
        for i, bullet in enumerate(self.player_bullets):
            for alien in self.aliens:
                if alien['alive'] and np.linalg.norm(bullet - alien['pos']) < 15:
                    # SFX: Alien explosion
                    alien['alive'] = False
                    bullets_to_remove.add(i)
                    self.explosions.append({'pos': alien['pos'].copy(), 'timer': 20, 'radius': 25})
                    self.score += 10
                    reward += 10
                    self.aliens_destroyed += 1
                    
                    if self.aliens_destroyed > 0 and self.aliens_destroyed % 5 == 0:
                        self.alien_speed += 0.05
                        self.alien_fire_rate += 0.001
                    break
        if bullets_to_remove:
            self.player_bullets = [b for i, b in enumerate(self.player_bullets) if i not in bullets_to_remove]
        
        # Alien bullets vs player
        hit_penalty = 0
        if self.player_hit_timer == 0:
            bullets_to_remove = set()
            for i, bullet in enumerate(self.alien_bullets):
                if np.linalg.norm(bullet - self.player_pos) < 20:
                    # SFX: Player hit
                    bullets_to_remove.add(i)
                    self.player_lives -= 1
                    hit_penalty -= 5
                    self.player_hit_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    self.explosions.append({'pos': self.player_pos.copy(), 'timer': 30, 'radius': 40})
                    break
            if bullets_to_remove:
                self.alien_bullets = [b for i, b in enumerate(self.alien_bullets) if i not in bullets_to_remove]
        
        return reward, hit_penalty

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
            "aliens_left": self.TOTAL_ALIENS - self.aliens_destroyed,
        }

    def _render_game(self):
        for alien in self.aliens:
            if alien['alive']:
                self._draw_alien(alien['pos'])

        if self.player_lives > 0:
            if self.player_hit_timer > 0 and self.steps % 10 < 5:
                pass  # Flash effect on hit
            else:
                self._draw_player(self.player_pos)
        
        for b in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, (int(b[0] - 2), int(b[1] - 5), 4, 10))
        for b in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_BULLET, (int(b[0] - 2), int(b[1] - 5), 4, 10))

        for exp in self.explosions:
            progress = exp['timer'] / (30.0 if exp['radius'] > 30 else 20.0)
            radius = int(exp['radius'] * (1.0 - progress**2))
            alpha = int(255 * progress)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color1 = self.COLOR_EXPLOSION_1 + (alpha,)
            color2 = self.COLOR_EXPLOSION_2 + (alpha,)
            pygame.draw.circle(s, color1, (radius, radius), radius)
            pygame.draw.circle(s, color2, (radius, radius), int(radius * 0.5))
            self.screen.blit(s, (int(exp['pos'][0] - radius), int(exp['pos'][1] - radius)))

    def _draw_player(self, pos):
        glow_surface = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (30, 30), 25)
        self.screen.blit(glow_surface, (int(pos[0]) - 30, int(pos[1]) - 30))

        points = [(pos[0], pos[1] - 15), (pos[0] - 15, pos[1] + 10), (pos[0] + 15, pos[1] + 10)]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _draw_alien(self, pos):
        p = (int(pos[0]), int(pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_ALIEN, (p[0] - 10, p[1] - 10, 20, 15))
        pygame.draw.rect(self.screen, self.COLOR_BG, (p[0] - 4, p[1] - 5, 8, 10))
        pygame.draw.rect(self.screen, self.COLOR_ALIEN, (p[0] - 8, p[1] + 5, 5, 5))
        pygame.draw.rect(self.screen, self.COLOR_ALIEN, (p[0] + 3, p[1] + 5, 5, 5))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.player_lives - 1):
            pos = (self.WIDTH - 30 - i * 25, 20)
            points = [(pos[0], pos[1] - 8), (pos[0] - 8, pos[1] + 5), (pos[0] + 8, pos[1] + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        if self.game_over:
            msg, color = ("VICTORY", self.COLOR_PLAYER) if self.win else ("GAME OVER", self.COLOR_ALIEN)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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