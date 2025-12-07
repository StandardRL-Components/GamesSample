
# Generated: 2025-08-27T17:01:56.852628
# Source Brief: brief_01406.md
# Brief Index: 1406

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block breaker. Clear all the blocks by bouncing the ball off your paddle. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
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
        
        # Colors (Bright and high-contrast)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (0, 200, 100), # Green
            20: (0, 150, 255), # Blue
            30: (255, 50, 100)  # Red
        }
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7
        self.MAX_STEPS = 2000
        self.INITIAL_BALLS = 3
        
        # Initialize state variables
        self.random_generator = None
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)
        else:
            # Use a default generator if no seed is provided
            if self.random_generator is None:
                self.random_generator = np.random.default_rng()

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = self.INITIAL_BALLS

        # Paddle state
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Ball state
        self._reset_ball()

        # Blocks
        self.blocks = self._create_blocks()
        
        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _create_blocks(self):
        blocks = []
        block_width = 58
        block_height = 20
        gap = 6
        rows = 5
        cols = 10
        total_block_width = cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                points = [10, 10, 20, 20, 30][r]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height
                )
                blocks.append({"rect": rect, "color": color, "points": points})
        return blocks
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        if not self.game_over:
            # 1. Handle player input
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

            # Launch ball
            if space_held and not self.ball_launched:
                self.ball_launched = True
                # sfx: launch_ball
                launch_angle = (self.random_generator.random() * 0.8 + 0.1) * math.pi # 18 to 162 degrees
                self.ball_vel = [-self.BALL_SPEED * math.cos(launch_angle), -self.BALL_SPEED * math.sin(launch_angle)]

            # 2. Update game logic
            self._update_ball()
            reward += self._handle_collisions()
            self._update_particles()
        
        # 3. Check for termination
        self.steps += 1
        terminated = self._check_termination()
        
        # Add terminal rewards
        if terminated:
            if self.win:
                reward += 100.0
            elif self.balls_left <= 0:
                reward += -50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_ball(self):
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
    
    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += 0.1
            # sfx: paddle_hit
            
            # Change angle based on where it hits the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BALL_SPEED * offset
            self.ball_vel[1] *= -1
            
            # Normalize velocity to maintain constant speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED
            
            # Ensure ball is above paddle to prevent sticking
            ball_rect.bottom = self.paddle.top

        # Ball lost
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            reward -= 2.0
            # sfx: ball_loss
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                reward += block["points"] / 10.0 # Scale reward to be in a reasonable range
                self.score += block["points"]
                # sfx: block_hit
                self._create_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                
                # Determine bounce direction
                # A simple vertical bounce is most common and stable for this genre
                self.ball_vel[1] *= -1
                break

        # Update ball position from rect after collision adjustments
        self.ball_pos[0] = ball_rect.centerx
        self.ball_pos[1] = ball_rect.centery
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.blocks:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.random_generator.random() * 2 * math.pi
            speed = self.random_generator.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.random_generator.random() * 3 + 2
            lifespan = 20 + self.random_generator.integers(10)
            self.particles.append({"pos": list(pos), "vel": vel, "size": size, "lifespan": lifespan, "color": color})
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["size"] -= 0.1
            if p["lifespan"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], width=2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw ball with glow
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 1.5)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS * 2, ball_center[1] - self.BALL_RADIUS * 2))
        pygame.draw.circle(self.screen, self.COLOR_BALL, ball_center, self.BALL_RADIUS)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), max(0, int(p["size"])))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        ball_text = self.font_ui.render("BALLS:", True, self.COLOR_UI_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 150, 10))
        for i in range(self.balls_left):
            pygame.draw.circle(self.screen, self.COLOR_BALL, (self.WIDTH - 70 + i * 20, 18), 6)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 128) if self.win else (255, 0, 0)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "win": self.win,
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful.")
    print("Initial info:", info)
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode ended.")
            break
            
    # Test a specific action sequence
    print("\nTesting specific action: move right and launch ball")
    env.reset()
    # Move right for a few frames
    for _ in range(5):
        env.step([4, 0, 0]) # Right, no space, no shift
    # Launch ball
    obs, reward, term, trunc, info = env.step([0, 1, 0]) # No move, space, no shift
    print(f"Launch step: Reward={reward:.2f}, Info={info}")
    assert info['score'] == 0 # Should not have hit a block yet
    # Let ball fly for a bit
    for i in range(50):
        obs, reward, term, trunc, info = env.step([0, 0, 0]) # No-op
        if reward > 0:
            print(f"Hit something on step {i+5+1}! Reward={reward:.2f}, Score={info['score']}")
            break
    
    env.close()
    print("\nAll tests passed.")