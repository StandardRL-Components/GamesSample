
# Generated: 2025-08-28T03:15:43.564289
# Source Brief: brief_01969.md
# Brief Index: 1969

        
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
        "Controls: Use ↑ and ↓ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style arcade game. Control the paddle to rebound the ball and destroy all the bricks. Don't let the ball pass your paddle!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # Colors
        self.COLOR_BG = (13, 13, 38) # Dark Blue
        self.COLOR_GRID = (26, 26, 64)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (0, 255, 255) # Cyan
        self.COLOR_BALL_GLOW = (0, 128, 128)
        self.COLOR_TEXT = (255, 255, 0) # Yellow
        self.BRICK_COLORS = [
            (255, 50, 50),   # Red
            (255, 150, 50),  # Orange
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (200, 50, 255),  # Magenta
        ]

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 80
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 500

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.base_ball_speed = 0
        self.bricks = []
        self.particles = []
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def _setup_bricks(self):
        self.bricks.clear()
        brick_width, brick_height = 40, 20
        rows, cols = 5, 12
        for i in range(rows):
            for j in range(cols):
                brick_x = 80 + j * (brick_width + 5)
                brick_y = 50 + i * (brick_height + 5)
                rect = pygame.Rect(brick_x, brick_y, brick_width, brick_height)
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append({"rect": rect, "color": color})

    def _reset_ball(self, new_life=False):
        self.paddle.y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.ball_pos = np.array([self.paddle.right + 20, self.paddle.centery], dtype=np.float64)
        
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        if self.np_random.random() < 0.5:
             angle += math.pi
        
        current_speed = self.base_ball_speed + 0.1 * (self.score // 100)
        self.ball_vel = np.array([math.cos(angle) * current_speed, math.sin(angle) * current_speed])

        if new_life:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.base_ball_speed = 6.0
        
        self.paddle = pygame.Rect(
            20, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        
        self.bricks = []
        self._setup_bricks()
        
        self.particles = []
        
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02  # Small time penalty
        
        if self.game_over:
            # If game is over, just return the current state without updates
            terminated = self.lives <= 0 or self.score >= self.WIN_SCORE
            return (
                self._get_observation(),
                0,
                terminated,
                False,
                self._get_info()
            )

        # 1. Handle player input
        movement = action[0]
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED
        
        # Clamp paddle position
        self.paddle.y = max(0, min(self.HEIGHT - self.PADDLE_HEIGHT, self.paddle.y))

        # 2. Update game logic
        self._update_ball()
        self._update_particles()
        
        # 3. Calculate rewards based on events
        # (Brick hit reward is handled in _update_ball)
        reward += getattr(self, "_step_reward", 0)
        self._step_reward = 0 # Reset step-specific reward

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
            else: # Lost by lives or timeout
                reward -= 100
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collisions
        if ball_rect.top <= 0:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1
        if ball_rect.right >= self.WIDTH:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
        if ball_rect.left <= 0: # Ball hits left wall behind paddle
             self.ball_pos[0] = self.BALL_RADIUS
             self.ball_vel[0] *= -1

        # Ball misses paddle
        if ball_rect.left > self.WIDTH: # Using right wall as reset boundary
            self._reset_ball(new_life=True)
            return

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            # Move ball out of paddle to prevent sticking
            self.ball_pos[0] = self.paddle.right + self.BALL_RADIUS
            
            # Reverse horizontal velocity
            self.ball_vel[0] *= -1
            
            # Add vertical velocity based on where it hit the paddle
            offset = (self.paddle.centery - self.ball_pos[1]) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] -= offset * 3.0 # Factor to control angle change
            
            # Re-normalize speed
            current_speed = np.linalg.norm(self.ball_vel)
            self.ball_vel = self.ball_vel / current_speed * (self.base_ball_speed + 0.1 * (self.score // 100))
            
            # Sound placeholder
            # pygame.mixer.Sound.play(paddle_hit_sound)

        # Brick collisions
        hit_brick_idx = ball_rect.collidelist([b["rect"] for b in self.bricks])
        if hit_brick_idx != -1:
            brick = self.bricks[hit_brick_idx]
            
            # Create particles
            self._create_particles(brick["rect"].center, brick["color"])
            
            # Determine bounce direction
            prev_ball_pos = self.ball_pos - self.ball_vel
            prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            
            # Simple but effective bounce logic
            if (prev_ball_rect.bottom <= brick["rect"].top or prev_ball_rect.top >= brick["rect"].bottom):
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

            # Remove brick and add score
            self.bricks.pop(hit_brick_idx)
            
            old_score_tier = self.score // 100
            self.score += 10
            new_score_tier = self.score // 100
            
            if new_score_tier > old_score_tier:
                # Increase ball speed
                new_speed = self.base_ball_speed + 0.1 * new_score_tier
                self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * new_speed

            self._step_reward = getattr(self, "_step_reward", 0) + 1.0
            
            # Sound placeholder
            # pygame.mixer.Sound.play(brick_break_sound)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        
    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS or not self.bricks

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            border_color = tuple(max(0, c - 50) for c in brick["color"])
            pygame.draw.rect(self.screen, border_color, brick["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            p_pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["lifespan"] / 25.0))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,4,4))
            self.screen.blit(temp_surf, p_pos_int)

        # UI
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))

        if self.game_over:
            win_lose_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE or not self.bricks else "GAME OVER"
            win_lose_text = self.font_large.render(win_lose_text_str, True, self.COLOR_TEXT)
            text_rect = win_lose_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(win_lose_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
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
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    clock = pygame.time.Clock()
    
    movement = 0 # 0=none, 1=up, 2=down
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP and movement == 1:
                    movement = 0
                elif event.key == pygame.K_DOWN and movement == 2:
                    movement = 0
    
        action = [movement, 0, 0]  # Movement, space, shift
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            done = True
            
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    env.close()