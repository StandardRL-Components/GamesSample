
# Generated: 2025-08-27T15:46:10.302923
# Source Brief: brief_01070.md
# Brief Index: 1070

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all the colored blocks by bouncing the ball off your paddle. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
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
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (200, 200, 220)
        self.BLOCK_COLORS = {
            10: (50, 205, 50),   # Green
            20: (30, 144, 255),  # Blue
            30: (220, 20, 60),   # Red
        }

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game constants
        self.WALL_THICKNESS = 10
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 12
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.MAX_BALL_SPEED_X = 5
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.block_data = None
        self.particles = None
        self.steps = None
        self.score = None
        self.balls_left = None
        self.game_over = None
        self.last_block_break_step = None
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_launched = False
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self._reset_ball()

        self._generate_blocks()
        
        self.particles = []
        self.last_block_break_step = -100 # For combo timer

        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def _generate_blocks(self):
        self.blocks = []
        self.block_data = []
        block_width = 58
        block_height = 20
        rows = 5
        cols = 10
        x_gap = 2
        y_gap = 2
        
        total_block_width = cols * (block_width + x_gap) - x_gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        point_values = [30, 30, 20, 20, 10]

        for r in range(rows):
            for c in range(cols):
                points = point_values[r]
                color = self.BLOCK_COLORS[points]
                
                x = start_x + c * (block_width + x_gap)
                y = start_y + r * (block_height + y_gap)
                
                block_rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append(block_rect)
                self.block_data.append({"color": color, "points": points})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.01  # Small penalty per step to encourage speed
        
        self._handle_input(movement, space_held)
        
        collision_reward = self._update_ball()
        reward += collision_reward
        
        self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        win_condition = not self.blocks
        lose_condition = self.balls_left <= 0
        timeout_condition = self.steps >= self.MAX_STEPS

        terminated = False
        if win_condition:
            reward += 100
            terminated = True
        elif lose_condition or timeout_condition:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle.left = max(self.WALL_THICKNESS, self.paddle.left)
        self.paddle.right = min(self.WIDTH - self.WALL_THICKNESS, self.paddle.right)

        # Ball launch
        if space_held and not self.ball_launched:
            self.ball_launched = True
            initial_dx = self.np_random.uniform(-2, 2)
            self.ball_vel = [initial_dx, -4]
            # Sound: Ball launch

        # Keep un-launched ball on paddle
        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

    def _update_ball(self):
        if not self.ball_launched:
            return 0
        
        event_reward = 0
        
        # Move ball
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= self.WALL_THICKNESS or self.ball.right >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            self.ball.left = max(self.ball.left, self.WALL_THICKNESS)
            self.ball.right = min(self.ball.right, self.WIDTH - self.WALL_THICKNESS)
            # Sound: Wall bounce
        if self.ball.top <= self.WALL_THICKNESS:
            self.ball_vel[1] *= -1
            self.ball.top = max(self.ball.top, self.WALL_THICKNESS)
            # Sound: Wall bounce

        # Bottom of screen (lose ball)
        if self.ball.top >= self.HEIGHT:
            self.balls_left -= 1
            event_reward -= 50
            if self.balls_left > 0:
                self._reset_ball()
            # Sound: Lose ball
            return event_reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            
            # Calculate reflection angle based on hit position
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] = offset * self.MAX_BALL_SPEED_X
            self.ball_vel[1] *= -1
            
            # Anti-stuck: ensure vertical velocity is not too low
            if abs(self.ball_vel[1]) < 1:
                self.ball_vel[1] = -1 * np.sign(self.ball_vel[1])
            # Sound: Paddle bounce

        # Block collision
        collided_index = self.ball.collidelist(self.blocks)
        if collided_index != -1:
            block_rect = self.blocks.pop(collided_index)
            block_info = self.block_data.pop(collided_index)
            
            # Add score and reward
            self.score += block_info["points"]
            event_reward += block_info["points"] / 10 # Scale reward to be smaller
            
            # Combo bonus
            if self.steps - self.last_block_break_step <= 15: # 0.5s combo window
                event_reward += 5
            self.last_block_break_step = self.steps

            self._create_particles(block_rect.center, block_info["color"])
            
            # Determine bounce direction (simple but effective)
            self.ball_vel[1] *= -1
            
            # Sound: Block break
            
        return event_reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), self.WALL_THICKNESS)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            color = self.block_data[i]["color"]
            pygame.draw.rect(self.screen, color, block)
            # Add a subtle highlight for depth
            highlight_color = tuple(min(255, c + 30) for c in color)
            pygame.draw.line(self.screen, highlight_color, block.topleft, block.topright, 2)
            pygame.draw.line(self.screen, highlight_color, block.topleft, block.bottomleft, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            
            # Using gfxdraw for anti-aliased circles
            x, y = int(p['pos'][0]), int(p['pos'][1])
            radius = int(p['life'] / p['max_life'] * 3)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, (*color, alpha))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with a glow effect
        x, y = int(self.ball.centerx), int(self.ball.centery)
        # Outer glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS + 2, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS + 2, (*self.COLOR_BALL, 50))
        # Inner ball
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))

        # Balls left
        balls_text = self.font_main.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        text_rect = balls_text.get_rect(topright=(self.WIDTH - self.WALL_THICKNESS - 10, self.WALL_THICKNESS + 5))
        self.screen.blit(balls_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, (200, 200, 200))
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
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