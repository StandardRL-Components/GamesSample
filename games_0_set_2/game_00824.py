
# Generated: 2025-08-27T14:53:56.036740
# Source Brief: brief_00824.md
# Brief Index: 824

        
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
        "Controls: ←→ to move the paddle. Keep the ball in play. Risky edge hits score more points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you control a paddle to keep a bouncing ball in play for 60 seconds. Maximize your score by risking close calls for bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 10
        self.BALL_SPEED = 7
        self.INITIAL_LIVES = 5
        self.MAX_STEPS = 1800  # 60 seconds * 30 fps
        self.RISKY_BOUNCE_THRESHOLD = 0.3 # 30% of paddle half-width

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PADDLE = (255, 0, 255) # Magenta
        self.COLOR_BALL = (0, 255, 255)   # Cyan
        self.COLOR_WALL = (255, 255, 255)
        self.COLOR_SPARK = (255, 255, 0)   # Yellow
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GLOW = (0, 150, 150)

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        # Initialize state variables
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        
        # This will be properly initialized in reset()
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_pos = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        
        # Initialize ball velocity
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Downwards
        self.ball_vel = [
            math.cos(angle) * self.BALL_SPEED,
            math.sin(angle) * self.BALL_SPEED,
        ]

        self.particles = []
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- 1. Update Paddle ---
        paddle_x_before = self.paddle_pos.x
        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos.x += self.PADDLE_SPEED
        
        self.paddle_pos.left = max(0, self.paddle_pos.left)
        self.paddle_pos.right = min(self.WIDTH, self.paddle_pos.right)

        # --- 2. Shaping Reward for Paddle Movement ---
        predicted_x = self._predict_ball_landing_x()
        if predicted_x is not None:
            dist_before = abs(paddle_x_before + self.PADDLE_WIDTH / 2 - predicted_x)
            dist_after = abs(self.paddle_pos.centerx - predicted_x)
            if dist_after > dist_before:
                reward -= 0.2 # Punish moving away from the predicted landing spot

        # --- 3. Update Ball ---
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # --- 4. Handle Collisions ---
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            self._create_particles(self.ball_pos)
            # sfx: wall_bounce.wav
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            self._create_particles(self.ball_pos)
            # sfx: wall_bounce.wav

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle_pos.colliderect(ball_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle_pos.top - self.BALL_RADIUS # Prevent sticking

            # Add spin based on hit location
            offset = (self.ball_pos[0] - self.paddle_pos.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2
            
            # Normalize ball speed after spin
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.BALL_SPEED

            # Risky bounce check
            is_risky = abs(offset) > (1.0 - self.RISKY_BOUNCE_THRESHOLD)
            if is_risky:
                bounce_reward = 2
                self.score += 15 # Bonus score for risky
                # sfx: risky_bounce.wav
            else:
                bounce_reward = 1
                self.score += 10
                # sfx: paddle_bounce.wav
            
            reward += bounce_reward
            self._create_particles(self.ball_pos, 30, is_risky)

        # Miss (bottom of screen)
        if self.ball_pos[1] >= self.HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            reward -= 5
            # sfx: lose_life.wav
            if self.lives > 0:
                # Reset ball
                self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
            else:
                self.game_over = True

        # --- 5. Update Particles ---
        self._update_particles()
        
        # --- 6. Update Game State ---
        self.steps += 1
        reward += 0.1 # Survival reward
        
        terminated = self._check_termination()

        # --- 7. Terminal Rewards ---
        if terminated:
            if self.steps >= self.MAX_STEPS: # Win condition
                reward += 50
                # sfx: win_game.wav
            if self.lives <= 0: # Lose condition
                reward -= 100
                # sfx: game_over.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _predict_ball_landing_x(self):
        # Only predict if ball is moving towards paddle
        if self.ball_vel[1] <= 0:
            return None

        # Simple projection without wall bounces for now, as it's a shaping heuristic
        y_to_travel = self.paddle_pos.top - self.ball_pos[1]
        if y_to_travel < 0: return None # Ball is already past the paddle
        
        time_to_impact = y_to_travel / self.ball_vel[1]
        
        # This simplified prediction ignores wall bounces for performance.
        # It's a heuristic, so it doesn't need to be perfect.
        predicted_x = self.ball_pos[0] + time_to_impact * self.ball_vel[0]
        
        # Simulate a single bounce off each wall
        if predicted_x < 0:
            predicted_x = -predicted_x
        elif predicted_x > self.WIDTH:
            predicted_x = self.WIDTH - (predicted_x - self.WIDTH)
            
        return predicted_x
    
    def _check_termination(self):
        return self.lives <= 0 or self.steps >= self.MAX_STEPS

    def _create_particles(self, pos, count=20, is_risky=False):
        color = self.COLOR_SPARK if is_risky else self.COLOR_WALL
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            size = int(p['lifespan'] / 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, self.COLOR_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, self.COLOR_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_pos, border_radius=3)

    def _render_ui(self):
        # Draw score
        score_text = self.font_large.render(f"{self.score:05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Draw lives
        heart_radius = 8
        for i in range(self.lives):
            x = self.WIDTH - 30 - i * (heart_radius * 2 + 10)
            y = 30
            pygame.gfxdraw.filled_circle(self.screen, x, y, heart_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, x, y, heart_radius, self.COLOR_PADDLE)

        # Draw timer
        time_left = max(0, (self.MAX_STEPS - self.steps) // 30)
        timer_text = self.font_large.render(f"{time_left}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Set up Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    # Game loop
    running = True
    while running and not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keyboard inputs to the MultiDiscrete action space
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        # These are unused in this game but required by the action space
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    env.close()