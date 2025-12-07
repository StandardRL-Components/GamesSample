
# Generated: 2025-08-27T18:30:49.425795
# Source Brief: brief_01852.md
# Brief Index: 1852

        
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
        "Controls: Use ← and → to move the paddle. "
        "Reflect the orb to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Reflect the enchanted orb with your crystal paddle to score points in this "
        "fast-paced, top-down fantasy arcade game. Don't let the orb pass you!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game parameters
    PADDLE_WIDTH = 120
    PADDLE_HEIGHT = 20
    PADDLE_SPEED = 10
    ORB_RADIUS = 10
    INITIAL_ORB_SPEED = 4.0
    ORB_SPEED_INCREMENT = 0.2
    MAX_ORB_SPEED = 12.0
    MAX_STEPS = 2000
    WIN_SCORE = 20
    MAX_LIVES = 3
    PADDLE_Y_POSITION = 360

    # Colors
    COLOR_BG_TOP = (15, 10, 40)
    COLOR_BG_BOTTOM = (48, 25, 90)
    COLOR_PADDLE = (120, 220, 255)
    COLOR_PADDLE_HIGHLIGHT = (200, 240, 255)
    COLOR_ORB = (255, 200, 80)
    COLOR_ORB_GLOW = (255, 180, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_PRIMARY = (255, 255, 220)
    COLOR_PARTICLE_SECONDARY = (255, 210, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Internal state variables
        self.paddle_rect = None
        self.orb_pos = None
        self.orb_vel = None
        self.particles = []
        self.orb_trail = []
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        
        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        # Paddle
        paddle_x = (self.screen.get_width() - self.PADDLE_WIDTH) / 2
        self.paddle_rect = pygame.Rect(paddle_x, self.PADDLE_Y_POSITION, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Orb
        self._reset_orb()
        
        # Effects
        self.particles = []
        self.orb_trail = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _reset_orb(self):
        """Resets the orb's position and velocity."""
        self.orb_pos = pygame.Vector2(self.screen.get_width() / 2, self.screen.get_height() / 3)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Downward cone
        self.orb_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.INITIAL_ORB_SPEED

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage faster play
        
        # Unpack factorized action
        movement = action[0]
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_orb()
        self._update_particles()
        
        # Collision detection and state changes
        reward += self._handle_collisions()
        
        # Check termination conditions
        self.steps += 1
        terminated = self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Win bonus
            elif self.lives <= 0:
                reward -= 50.0 # Loss penalty

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle_rect.x = max(0, min(self.paddle_rect.x, self.screen.get_width() - self.PADDLE_WIDTH))

    def _update_orb(self):
        self.orb_pos += self.orb_vel
        self.orb_trail.append(self.orb_pos.copy())
        if len(self.orb_trail) > 15:
            self.orb_trail.pop(0)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _handle_collisions(self):
        reward = 0.0
        orb_rect = pygame.Rect(self.orb_pos.x - self.ORB_RADIUS, self.orb_pos.y - self.ORB_RADIUS, self.ORB_RADIUS * 2, self.ORB_RADIUS * 2)

        # Wall collisions
        if self.orb_pos.x <= self.ORB_RADIUS or self.orb_pos.x >= self.screen.get_width() - self.ORB_RADIUS:
            self.orb_vel.x *= -1
            self.orb_pos.x = max(self.ORB_RADIUS, min(self.orb_pos.x, self.screen.get_width() - self.ORB_RADIUS))
            self._create_particles(self.orb_pos, 5, self.COLOR_PARTICLE_PRIMARY)
            # sfx: wall_bounce.wav
        
        if self.orb_pos.y <= self.ORB_RADIUS:
            self.orb_vel.y *= -1
            self.orb_pos.y = max(self.ORB_RADIUS, self.orb_pos.y)
            self._create_particles(self.orb_pos, 5, self.COLOR_PARTICLE_PRIMARY)
            # sfx: wall_bounce.wav

        # Paddle collision
        if orb_rect.colliderect(self.paddle_rect) and self.orb_vel.y > 0:
            self.orb_vel.y *= -1
            
            # Add horizontal influence based on hit location
            offset = (orb_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.orb_vel.x += offset * 2.0
            
            # Increase speed
            current_speed = self.orb_vel.length()
            if current_speed < self.MAX_ORB_SPEED:
                self.orb_vel.scale_to_length(current_speed + self.ORB_SPEED_INCREMENT)
            
            self.score += 1
            reward += 1.0
            self._create_particles(self.orb_pos, 20, self.COLOR_PARTICLE_SECONDARY)
            # sfx: paddle_hit.wav

        # Bottom edge (miss)
        if self.orb_pos.y > self.screen.get_height() + self.ORB_RADIUS:
            self.lives -= 1
            reward -= 10.0
            self._reset_orb()
            # sfx: life_lost.wav
        
        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        # --- Render all game elements ---
        self._render_background()
        self._render_trail()
        self._render_particles()
        self._render_paddle()
        self._render_orb()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.screen.get_height()):
            ratio = y / self.screen.get_height()
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen.get_width(), y))

    def _render_trail(self):
        for i, pos in enumerate(self.orb_trail):
            alpha = int(255 * (i / len(self.orb_trail)))
            radius = int(self.ORB_RADIUS * 0.5 * (i / len(self.orb_trail)))
            if radius > 0:
                glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_ORB_GLOW, alpha // 2), (radius, radius), radius)
                self.screen.blit(glow_surf, (int(pos.x - radius), int(pos.y - radius)))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_paddle(self):
        # Base shape
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)
        
        # Crystal highlight facets
        p = self.paddle_rect
        points1 = [(p.left + 5, p.top + 5), (p.centerx - 10, p.top + 5), (p.centerx, p.centery), (p.left + 15, p.centery)]
        points2 = [(p.centerx + 10, p.top + 5), (p.right - 5, p.top + 5), (p.right - 15, p.centery), (p.centerx, p.centery)]
        pygame.gfxdraw.aapolygon(self.screen, points1, self.COLOR_PADDLE_HIGHLIGHT)
        pygame.gfxdraw.filled_polygon(self.screen, points1, (*self.COLOR_PADDLE_HIGHLIGHT, 100))
        pygame.gfxdraw.aapolygon(self.screen, points2, self.COLOR_PADDLE_HIGHLIGHT)
        pygame.gfxdraw.filled_polygon(self.screen, points2, (*self.COLOR_PADDLE_HIGHLIGHT, 100))

    def _render_orb(self):
        # Glow effect
        glow_radius = int(self.ORB_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_ORB_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.orb_pos.x - glow_radius), int(self.orb_pos.y - glow_radius)))
        
        # Main orb
        pygame.gfxdraw.aacircle(self.screen, int(self.orb_pos.x), int(self.orb_pos.y), self.ORB_RADIUS, self.COLOR_ORB)
        pygame.gfxdraw.filled_circle(self.screen, int(self.orb_pos.x), int(self.orb_pos.y), self.ORB_RADIUS, self.COLOR_ORB)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_large.render(f"Lives: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.screen.get_width() - lives_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_ORB)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_TEXT)

            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Game loop
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Set movement action
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                 raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((640, 400))
            pygame.display.set_caption("Crystal Paddle")
            display_surf.blit(surf, (0, 0))

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset() # Auto-reset
            pygame.time.wait(2000) # Pause before restarting
            
        env.clock.tick(30) # Run at 30 FPS

    env.close()