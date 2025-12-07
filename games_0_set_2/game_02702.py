
# Generated: 2025-08-27T21:09:57.768049
# Source Brief: brief_02702.md
# Brief Index: 2702

        
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
        "Controls: ←→ to move the paddle. Clear all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block-breaking game. Use the paddle to bounce the ball and "
        "strategically clear the grid of blocks. Clear all 100 blocks to win, "
        "but lose all 3 balls and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # Initialize state variables
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_remaining = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.last_block_break_step = -10
        
        # Game constants
        self._setup_constants()
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()

    def _setup_constants(self):
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PADDLE = (0, 200, 255)
        self.COLOR_PADDLE_OUTLINE = (100, 240, 255)
        self.COLOR_BALL = (255, 180, 0)
        self.COLOR_BALL_GLOW = (100, 70, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WALLS = (80, 80, 100)
        
        # Paddle properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        
        # Ball properties
        self.BALL_RADIUS = 7
        self.BALL_INITIAL_SPEED = 5.0
        self.BALL_MAX_SPEED = 10.0

        # Block properties
        self.BLOCK_ROWS = 10
        self.BLOCK_COLS = 10
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 15
        self.BLOCK_SPACING = 2
        self.BLOCK_OFFSET_X = (self.SCREEN_WIDTH - (self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING))) // 2
        self.BLOCK_OFFSET_Y = 50

        # Episode termination
        self.MAX_STEPS = 2000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_remaining = 3
        self.last_block_break_step = -10
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._create_blocks()
        self._reset_ball()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = self.BLOCK_OFFSET_X + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_OFFSET_Y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                
                # Color gradient from red (bottom) to green (top)
                hue = 120 * (1 - i / self.BLOCK_ROWS)
                color = pygame.Color(0)
                color.hsla = (hue, 100, 50, 100)
                
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color})

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards direction
        self.ball_vel = [
            self.BALL_INITIAL_SPEED * math.cos(angle),
            self.BALL_INITIAL_SPEED * math.sin(angle)
        ]

    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.02  # Per-step penalty to encourage efficiency
        
        # 1. Handle player input
        self._handle_input(movement)
        
        # 2. Update game objects
        self._update_ball()
        self._update_particles()
        
        # 3. Handle collisions and calculate rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        # 4. Check for win/loss conditions
        if not self.blocks:
            self.win = True
            reward += 100  # Victory bonus
        
        # 5. Check for termination
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        if self.game_over and not self.win:
            reward -= 50 # Game over penalty
        
        # 6. Update step counter
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Keep paddle within bounds
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Walls
        if ball_rect.left <= 0:
            self.ball_vel[0] *= -1
            ball_rect.left = 0
            # sfx: wall_bounce.wav
        if ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.right = self.SCREEN_WIDTH
            # sfx: wall_bounce.wav
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            # sfx: wall_bounce.wav

        # Bottom wall (lose a ball)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.balls_remaining -= 1
            reward -= 2
            # sfx: lose_life.wav
            if self.balls_remaining > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit.wav
            self.ball_vel[1] *= -1
            # Move ball out of paddle to prevent sticking
            ball_rect.bottom = self.paddle.top
            
            # Change horizontal velocity based on impact point
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BALL_INITIAL_SPEED * offset * 1.5
            
            # Normalize speed to prevent it from getting too fast/slow
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_INITIAL_SPEED
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_INITIAL_SPEED


        # Blocks
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # sfx: block_break.wav
                self.blocks.remove(block)
                reward += 1
                self.score += 10
                
                # Combo reward
                if self.steps - self.last_block_break_step <= 3:
                    reward += 5
                    self.score += 20 # Bonus score for combo
                self.last_block_break_step = self.steps

                # Create particles
                self._create_particles(block["rect"].center, block["color"])

                # Reflect ball
                # A simple approximation: check if collision is more horizontal or vertical
                prev_ball_rect = pygame.Rect(
                    (self.ball_pos[0] - self.ball_vel[0]) - self.BALL_RADIUS,
                    (self.ball_pos[1] - self.ball_vel[1]) - self.BALL_RADIUS,
                    self.BALL_RADIUS * 2,
                    self.BALL_RADIUS * 2,
                )
                
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                break # Only break one block per frame

        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return reward
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.rng.integers(10, 25)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, width=2, border_radius=5)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 25))
            color_with_alpha = p["color"][:3] + (alpha,)
            temp_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls remaining
        balls_text = self.font_ui.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over or self.win:
            message = "YOU WIN!" if self.win else "GAME OVER"
            msg_surf = self.font_msg.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "blocks_remaining": len(self.blocks),
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


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a window for manual play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()

    while not terminated:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display window
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Control the speed of manual play
        
    env.close()
    pygame.quit()