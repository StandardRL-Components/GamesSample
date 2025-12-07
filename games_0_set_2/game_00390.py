
# Generated: 2025-08-27T13:29:59.558098
# Source Brief: brief_00390.md
# Brief Index: 390

        
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
        "Controls: Use ← and → to move the paddle left and right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A modern, vibrant Brick Breaker. Clear all bricks to win, but lose the ball 3 times and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MIN_SPEED = 6
        self.BALL_MAX_SPEED = 12
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3

        # --- Colors ---
        self.COLOR_BG_TOP = (15, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 50, 80)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 100, 50)
        self.COLOR_WALL = (100, 110, 140)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = {
            10: (50, 200, 50),   # Green
            25: (50, 100, 255),  # Blue
            50: (220, 50, 50),   # Red
        }

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_paddle_hit_factor = 0.0

        # --- Initialize state ---
        self.reset()
        
        # --- Validate implementation ---
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.particles = []
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._generate_bricks()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Time penalty to encourage efficiency
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # --- Update Game Logic ---
            self._update_paddle(movement)
            reward += self._update_ball()
            self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        win = len(self.bricks) == 0
        loss = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS
        
        terminated = win or loss or timeout
        
        if terminated and not self.game_over:
            if win:
                reward += 100.0
            if loss:
                reward -= 100.0
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [math.cos(angle) * self.BALL_MIN_SPEED, math.sin(angle) * self.BALL_MIN_SPEED]
        self.last_paddle_hit_factor = 0.0

    def _generate_bricks(self):
        self.bricks = []
        brick_rows = 5
        brick_cols = 15
        brick_width = self.WIDTH // brick_cols
        brick_height = 20
        
        for r in range(brick_rows):
            for c in range(brick_cols):
                point_val = [50, 25, 25, 10, 10][r] # Red, Blue, Blue, Green, Green
                color = self.BRICK_COLORS[point_val]
                brick = pygame.Rect(
                    c * brick_width,
                    r * brick_height + 50,
                    brick_width - 1,
                    brick_height - 1
                )
                self.bricks.append({"rect": brick, "points": point_val, "color": color})

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

    def _update_ball(self):
        reward = 0.0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.ball_pos[0], self.WIDTH - self.BALL_RADIUS))
            # sfx: wall_bounce.wav
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])
            # sfx: wall_bounce.wav
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            
            # Calculate hit factor and adjust horizontal velocity
            self.last_paddle_hit_factor = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.last_paddle_hit_factor = max(-1, min(1, self.last_paddle_hit_factor))
            self.ball_vel[0] += self.last_paddle_hit_factor * 4
            
            # Normalize speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > self.BALL_MAX_SPEED:
                self.ball_vel = [v * self.BALL_MAX_SPEED / speed for v in self.ball_vel]
            if speed < self.BALL_MIN_SPEED:
                self.ball_vel = [v * self.BALL_MIN_SPEED / speed for v in self.ball_vel]
            # sfx: paddle_hit.wav
        
        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick["rect"]):
                self.bricks.remove(brick)
                self.ball_vel[1] *= -1
                
                # Add reward for breaking brick
                reward += brick["points"]
                self.score += brick["points"]
                
                # Add risk/reward based on last paddle hit
                hit_abs = abs(self.last_paddle_hit_factor)
                if hit_abs > 0.75: # Risky edge hit
                    reward += 0.1
                elif hit_abs < 0.25: # Safe center hit
                    reward -= 0.2
                
                self._create_particles(brick["rect"].center, brick["color"])
                # sfx: brick_break.wav
                break

        # Ball lost
        if self.ball_pos[1] > self.HEIGHT:
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
                # sfx: life_lost.wav
            else:
                self.game_over = True
                # sfx: game_over.wav

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            particle = {
                "pos": list(pos),
                "vel": [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                "life": self.np_random.uniform(15, 30),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["vel"][1] += 0.1 # Gravity
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks)
        }
        
    def _render_all(self):
        # --- Background Gradient ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # --- Game Elements ---
        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 10)))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill((*p["color"], alpha))
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        glow_radius = self.BALL_RADIUS * 2
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (ball_x - glow_radius, ball_y - glow_radius))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # --- UI ---
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            message = "YOU WIN!" if len(self.bricks) == 0 else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to see the game being played
    render_mode = "human" # or "rgb_array"
    
    if render_mode == "human":
        # For human rendering, we need a real display
        env = GameEnv(render_mode="rgb_array")
        pygame.display.set_caption("Brick Breaker")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    else:
        env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # This maps keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_LEFT: [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }
    
    action = [0, 0, 0] # Default no-op action
    
    frame_count = 0
    start_time = time.time()

    while not done:
        if render_mode == "human":
            # Event handling for human play
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        action = key_to_action[event.key]
                if event.type == pygame.KEYUP:
                    if event.key in key_to_action:
                        action = [0, 0, 0] # No-op on key release
        else:
            # For rgb_array mode, just sample random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render_mode == "human":
            # Blit the observation from the env to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit to 30 FPS

        frame_count += 1
        if info.get('score') != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Lives: {info['lives']}")

    end_time = time.time()
    duration = end_time - start_time
    fps = frame_count / duration if duration > 0 else 0
    print(f"\nGame Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Steps: {info['steps']}")
    print(f"Avg FPS: {fps:.2f}")

    env.close()