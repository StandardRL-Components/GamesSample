
# Generated: 2025-08-28T03:17:05.280435
# Source Brief: brief_04876.md
# Brief Index: 4876

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball. "
        "Hold shift to temporarily speed up the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risky plays are rewarded and safe plays are penalized."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (0, 200, 100), # Green
            20: (100, 150, 255), # Blue
            30: (255, 80, 80) # Red
        }

        # Game element properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_BASE_SPEED_MAGNITUDE = 6
        self.BALL_BOOST_SPEED_MAGNITUDE = 10
        self.BOOST_DURATION = 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.boost_timer = None

        # --- Initialize state ---
        self.reset()
        
        # --- Validate implementation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.game_over = False
        self.boost_timer = 0
        self.particles = []

        # Paddle state
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball state
        self.ball_launched = False
        self._reset_ball()

        # Block layout
        self.blocks = []
        block_width = 50
        block_height = 20
        num_cols = 10
        num_rows = 5
        gap = 5
        total_block_width = num_cols * (block_width + gap) - gap
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        start_y = 50

        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * (block_width + gap)
                y = start_y + row * (block_height + gap)
                
                if row < 1:
                    points = 30 # Red
                elif row < 3:
                    points = 20 # Blue
                else:
                    points = 10 # Green
                
                color = self.BLOCK_COLORS[points]
                block_rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({"rect": block_rect, "points": points, "color": color})
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.boost_timer = 0

    def step(self, action):
        reward = -0.02  # Small penalty per step to encourage speed
        
        if self.game_over:
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        self.steps += 1
        
        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        self._handle_paddle_movement(movement)
        self._handle_ball_launch(space_held)
        self._handle_boost(shift_held)

        # --- Update Game State ---
        if self.ball_launched:
            ball_events_reward = self._update_ball()
            reward += ball_events_reward
        else:
            # Ball follows paddle before launch
            self.ball_pos.x = self.paddle.centerx
        
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        if len(self.blocks) == 0:
            reward += 100
            self.game_over = True
            # Sound: Win jingle
        elif self.balls_left <= 0:
            reward += -100
            self.game_over = True
            # Sound: Game over sound
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_paddle_movement(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

    def _handle_ball_launch(self, space_held):
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # Sound: Ball launch
            # Give it a slight random initial x direction
            initial_x_vel = self.np_random.uniform(-0.5, 0.5)
            self.ball_vel = pygame.Vector2(initial_x_vel, -1)
            self.ball_vel.scale_to_length(self.BALL_BASE_SPEED_MAGNITUDE)

    def _handle_boost(self, shift_held):
        if shift_held and self.ball_launched and self.boost_timer == 0:
            self.boost_timer = self.BOOST_DURATION
            # Sound: Boost activate
        
        if self.boost_timer > 0:
            self.ball_vel.scale_to_length(self.BALL_BOOST_SPEED_MAGNITUDE)
            self.boost_timer -= 1
        elif self.ball_launched: # Return to base speed after boost
            # Check to avoid division by zero if vel is (0,0)
            if self.ball_vel.length() > 0:
                self.ball_vel.scale_to_length(self.BALL_BASE_SPEED_MAGNITUDE)

    def _update_ball(self):
        step_reward = 0
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.SCREEN_WIDTH - self.BALL_RADIUS))
            # Sound: Wall bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # Sound: Wall bounce

        # Lost ball
        if self.ball_pos.y > self.SCREEN_HEIGHT:
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
                # Sound: Lose a life
            # The terminal penalty is applied in step()

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            step_reward += 0.1 # Base reward for deflection
            # Sound: Paddle bounce
            self.ball_vel.y *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = offset * self.BALL_BASE_SPEED_MAGNITUDE * 0.8 # Max influence
            
            # Risky play check
            if abs(offset) > 0.6: # Hit with the outer 40% of the paddle
                step_reward += -0.1

            # Prevent ball from getting stuck inside paddle
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

        # Block collisions
        for block in self.blocks[:]:
            if block["rect"].colliderect(ball_rect):
                step_reward += block["points"]
                self.score += block["points"]
                self.blocks.remove(block)
                # Sound: Block break
                self._create_particles(block["rect"].center, block["color"])

                # Determine collision side to correctly reflect
                # Simplified reflection: just reverse y-velocity
                self.ball_vel.y *= -1
                break # Only break one block per frame
        
        # Anti-softlock: Ensure minimum speed
        if self.ball_launched and self.ball_vel.length() < 0.1:
            self.ball_vel.y = -1 # Nudge it
            self.ball_vel.scale_to_length(self.BALL_BASE_SPEED_MAGNITUDE)

        return step_reward
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_all(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        
        # Render game elements
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_paddle(self):
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE)
        pygame.gfxdraw.rectangle(self.screen, self.paddle, (200, 200, 200))

    def _render_ball(self):
        x, y = int(self.ball_pos.x), int(self.ball_pos.y)
        radius = self.BALL_RADIUS
        
        # Glow effect
        glow_radius = int(radius * 1.8) if self.boost_timer > 0 else int(radius * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, self.COLOR_BALL_GLOW)
        
        # Main ball
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_BALL)

    def _render_blocks(self):
        for block in self.blocks:
            pygame.gfxdraw.box(self.screen, block["rect"], block["color"])
            # Add a slight border for definition
            border_color = tuple(max(0, c - 40) for c in block["color"])
            pygame.gfxdraw.rectangle(self.screen, block["rect"], border_color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 20.0))))
            color = p["color"] + (alpha,)
            size = max(1, int(p["lifespan"] / 5))
            rect = pygame.Rect(p["pos"].x - size//2, p["pos"].y - size//2, size, size)
            pygame.gfxdraw.box(self.screen, rect, color)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            x = self.SCREEN_WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        message = "YOU WIN!" if len(self.blocks) == 0 else "GAME OVER"
        text = self.font_game_over.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Risky Block Breaker")
    clock = pygame.time.Clock()

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # Convert the observation (H, W, C) to a Pygame Surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling and FPS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the auto_advance rate

    print(f"Game Over! Final Info: {info}")
    env.close()