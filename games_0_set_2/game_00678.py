
# Generated: 2025-08-27T14:24:48.733717
# Source Brief: brief_00678.md
# Brief Index: 678

        
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
    """
    A Gymnasium environment for a haunted pong survival game.

    The player controls a paddle at the bottom of the screen and must survive for
    60 seconds against a ghostly, increasingly fast ball. Losing all three
    lives or running out of time ends the episode. The game prioritizes visual
    polish with particle effects, smooth animations, and a gothic horror theme.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ←→ to move the paddle. Survive the ghost ball for 60 seconds."
    )
    game_description = (
        "Survive the onslaught of a ghostly pong ball in a top-down haunted "
        "mansion for 60 seconds. The ball gets faster and more erratic over time."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 100  # Brief specified 100fps
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Gameplay Constants
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 5
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 2.0
    BALL_SPEED_INCREASE_INTERVAL = 10 * FPS  # Every 10 seconds
    BALL_SPEED_INCREASE_AMOUNT = 0.5
    MAX_LIVES = 3
    BOUNCE_ANGLE_VARIATION = 7.5  # Degrees of random variation on bounce

    # Colors (Gothic Horror Theme)
    COLOR_BG = (15, 10, 20)
    COLOR_WOOD_DARK = (40, 30, 25)
    COLOR_WOOD_LIGHT = (55, 45, 40)
    COLOR_PADDLE = (70, 50, 40)
    COLOR_PADDLE_BORDER = (90, 70, 60)
    COLOR_BALL = (220, 230, 255)
    COLOR_BALL_GLOW = (150, 180, 255, 50) # RGBA
    COLOR_PARTICLE = (200, 210, 240)
    COLOR_UI_TEXT = (180, 220, 180)
    COLOR_HEART = (200, 40, 40)
    COLOR_FLICKER = (100, 80, 0, 10) # RGBA

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.particles = []
        self.ball_trail = []
        self.flicker_radius = 0

        # This will be initialized in reset()
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False

        # Player paddle
        self.paddle_pos = pygame.math.Vector2(
            self.SCREEN_WIDTH / 2 - self.PADDLE_WIDTH / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2,
        )

        # Ball state
        self.ball_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 3)
        initial_angle = self.np_random.uniform(225, 315) # Downward direction
        self.ball_vel = pygame.math.Vector2(1, 0).rotate(initial_angle) * self.INITIAL_BALL_SPEED

        # Effects
        self.particles = []
        self.ball_trail = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        reward = 0

        # Penalty for moving when ball is moving away
        if self.ball_vel.y < 0 and movement in [3, 4]:
            reward -= 0.02

        if movement == 3:  # Move left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Move right
            self.paddle_pos.x += self.PADDLE_SPEED

        # Clamp paddle to screen
        self.paddle_pos.x = max(0, min(self.paddle_pos.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # --- Game Logic ---
        self.steps += 1
        self.score = self.steps // self.FPS # Score is seconds survived

        # Update ball position
        self.ball_pos += self.ball_vel

        # --- Collision Detection ---
        paddle_rect = pygame.Rect(self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball vs Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_vel.rotate_ip(self.np_random.uniform(-self.BOUNCE_ANGLE_VARIATION, self.BOUNCE_ANGLE_VARIATION))
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.SCREEN_WIDTH - self.BALL_RADIUS))
            self._create_particles(self.ball_pos, 10)
            # sfx: wall_bounce.wav

        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_vel.rotate_ip(self.np_random.uniform(-self.BOUNCE_ANGLE_VARIATION, self.BOUNCE_ANGLE_VARIATION))
            self.ball_pos.y = self.BALL_RADIUS
            self._create_particles(self.ball_pos, 10)
            # sfx: wall_bounce.wav

        # Ball vs Paddle
        if self.ball_vel.y > 0 and paddle_rect.collidepoint(self.ball_pos.x, self.ball_pos.y + self.BALL_RADIUS):
            reward += 1.0 # Successful deflection
            self.ball_vel.y *= -1
            
            # Influence bounce angle based on hit location
            hit_offset = (self.ball_pos.x - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += hit_offset * 2.0
            
            self.ball_vel.rotate_ip(self.np_random.uniform(-self.BOUNCE_ANGLE_VARIATION, self.BOUNCE_ANGLE_VARIATION))
            self.ball_pos.y = paddle_rect.top - self.BALL_RADIUS
            self._create_particles(self.ball_pos, 20, (255, 255, 100))
            # sfx: paddle_hit.wav

        # Ball vs Bottom Wall (Lose Life)
        if self.ball_pos.y - self.BALL_RADIUS > self.SCREEN_HEIGHT:
            self.lives -= 1
            # sfx: lose_life.wav
            if self.lives <= 0:
                self.game_over = True
                reward -= 100.0 # Lost all lives
                # sfx: game_over.wav
            else:
                # Reset ball
                self.ball_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 3)
                initial_angle = self.np_random.uniform(225, 315)
                current_speed = self.ball_vel.length()
                self.ball_vel = pygame.math.Vector2(1, 0).rotate(initial_angle) * current_speed

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.BALL_SPEED_INCREASE_INTERVAL == 0:
            current_speed = self.ball_vel.length()
            new_speed = min(current_speed + self.BALL_SPEED_INCREASE_AMOUNT, 10) # Cap speed
            if new_speed > current_speed:
                self.ball_vel.scale_to_length(new_speed)
                # sfx: speed_up.wav

        # --- Termination Check ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.game_over and self.steps >= self.MAX_STEPS:
            reward += 100.0 # Survived!
            self.game_over = True # Set game over to stop logic on next step
            # sfx: victory.wav

        # Survival reward
        if not terminated:
            reward += 0.01

        # --- Update Effects ---
        self._update_effects()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Clear screen and render all elements
        self._render_background()
        self._render_effects()
        self._render_game_objects()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    # --- Rendering Methods ---

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Procedural wood floor
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_WOOD_DARK, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_WIDTH, 80):
             pygame.draw.line(self.screen, self.COLOR_WOOD_LIGHT, (i, 0), (i, self.SCREEN_HEIGHT), 2)
        
        # Flickering light source
        self.flicker_radius = 300 + math.sin(self.steps * 0.05) * 10 + self.np_random.uniform(-5, 5)
        flicker_surf = pygame.Surface((self.flicker_radius * 2, self.flicker_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(flicker_surf, self.COLOR_FLICKER, (self.flicker_radius, self.flicker_radius), self.flicker_radius)
        self.screen.blit(flicker_surf, (self.SCREEN_WIDTH/2 - self.flicker_radius, self.SCREEN_HEIGHT/2 - self.flicker_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_game_objects(self):
        # Draw paddle
        paddle_rect = pygame.Rect(self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_BORDER, paddle_rect, width=2, border_radius=3)
        
        # Draw ball glow
        glow_radius = int(self.BALL_RADIUS * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_effects(self):
        # Draw ball trail
        for i, (pos, alpha) in enumerate(self.ball_trail):
            color = (*self.COLOR_BALL, int(alpha))
            trail_surf = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, color, (self.BALL_RADIUS, self.BALL_RADIUS), self.BALL_RADIUS - i//2)
            self.screen.blit(trail_surf, (int(pos.x - self.BALL_RADIUS), int(pos.y - self.BALL_RADIUS)))

        # Draw particles
        for p in self.particles:
            p_color = (*p['color'], int(p['alpha']))
            p_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, p_color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(p_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score:04d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:05.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Lives (Hearts)
        for i in range(self.lives):
            self._draw_heart(self.SCREEN_WIDTH - 30 - i * 35, 50)

    def _draw_heart(self, x, y):
        # A simple heart shape using polygons
        points = [
            (x, y + 5), (x - 12, y - 5), (x - 8, y - 12), (x, y - 5),
            (x + 8, y - 12), (x + 12, y - 5)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_HEART, points)

    # --- Effect Management ---

    def _create_particles(self, position, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.math.Vector2(1, 0).rotate(angle) * speed
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'radius': int(self.np_random.uniform(2, 5)),
                'alpha': 255,
                'decay': self.np_random.uniform(3, 8),
                'color': color
            })

    def _update_effects(self):
        # Update ball trail
        self.ball_trail.insert(0, (self.ball_pos.copy(), 100))
        if len(self.ball_trail) > 10:
            self.ball_trail.pop()
        for i in range(len(self.ball_trail)):
            pos, alpha = self.ball_trail[i]
            self.ball_trail[i] = (pos, alpha * 0.9)

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['alpha'] -= p['decay']
        self.particles = [p for p in self.particles if p['alpha'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == self.observation_space.shape
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space after reset
        test_obs_reset = self._get_observation()
        assert test_obs_reset.shape == self.observation_space.shape
        assert test_obs_reset.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == self.observation_space.shape
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# Example usage for human play
if __name__ == '__main__':
    # Set this to run the environment in a window
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'dummy' etc.

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 1000))
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Ghostly Pong Survival")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0.0

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        # The other actions are unused in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height) but numpy uses (height, width, channels)
        # We need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS) # Match environment's internal clock

        if terminated:
            print(f"Game Over!")
            print(f"Final Score: {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")

    env.close()