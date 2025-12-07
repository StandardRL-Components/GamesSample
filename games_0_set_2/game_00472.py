
# Generated: 2025-08-27T13:44:49.644172
# Source Brief: brief_00472.md
# Brief Index: 472

        
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
        "A fast-paced, grid-based block breaker. Clear all the blocks to win. "
        "Hitting the ball with the edges of your paddle gives a score bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3600 # 2 minutes at 30fps

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (0, 255, 255)
        self.COLOR_BALL_GLOW = (0, 150, 150)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_BORDER = (40, 40, 60)
        self.BLOCK_COLORS = {
            10: (50, 200, 50),   # Green
            20: (50, 100, 200),  # Blue
            30: (200, 50, 50),   # Red
            40: (200, 200, 50),  # Yellow
        }
        
        # Physics & Sizes
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED_Y = 6
        self.BALL_MAX_SPEED_X = 7

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("consolas", 40, bold=True)
        
        # --- Game State Attributes ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.lives = 0
        self.blocks = []
        self.particles = []
        
        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.particles.clear()
        
        # Paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self._reset_ball()
        
        # Blocks
        self._generate_blocks()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        self.blocks.clear()
        block_width = 58
        block_height = 20
        gap = 5
        rows, cols = 5, 10
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) // 2
        start_y = 50
        
        point_values = [40, 40, 30, 30, 20, 20, 10, 10, 10, 10]
        
        for r in range(rows):
            for c in range(cols):
                points = point_values[c] if r < 2 else point_values[c-2] if r < 4 else point_values[c-4]
                points = random.choice(list(self.BLOCK_COLORS.keys())) # More random layout
                color = self.BLOCK_COLORS[points]
                
                block_rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "color": color, "points": points})

    def step(self, action):
        reward = -0.01  # Small time penalty to encourage action
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # --- Update game logic ---
            self._handle_input(movement, space_held)
            self._update_ball()
            
            # Collision checks
            reward += self._check_paddle_collision()
            block_reward, block_broken = self._check_block_collisions()
            reward += block_reward
            
            if self._check_wall_collisions():
                # Ball hit bottom wall
                self.lives -= 1
                # sound: lose_life.wav
                if self.lives > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    reward -= 100 # Terminal penalty
        
        # Update particles
        self._update_particles()
        
        # Check termination conditions
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.blocks and not self.game_over: # Win condition
            self.game_over = True
            terminated = True
            reward += 100 # Terminal reward
        
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
        
        self.paddle.x = max(5, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - 5))

        # Ball launch
        if space_held and not self.ball_launched:
            self.ball_launched = True
            initial_vx = (self.np_random.random() - 0.5) * 2 # Random direction
            self.ball_vel = [initial_vx, -self.BALL_MAX_SPEED_Y]
            # sound: launch.wav

        # If ball isn't launched, it follows the paddle
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle.centerx

    def _update_ball(self):
        if self.ball_launched:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
    
    def _check_wall_collisions(self):
        # Left/Right walls
        if self.ball_pos[0] <= self.BALL_RADIUS + 5 or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS - 5:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS + 5, min(self.ball_pos[0], self.WIDTH - self.BALL_RADIUS - 5))
            # sound: bounce_wall.wav

        # Top wall
        if self.ball_pos[1] <= self.BALL_RADIUS + 5:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS + 5
            # sound: bounce_wall.wav

        # Bottom wall (loss of life)
        if self.ball_pos[1] >= self.HEIGHT:
            return True
        return False

    def _check_paddle_collision(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            # sound: bounce_paddle.wav
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

            # Influence X velocity based on hit location
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.BALL_MAX_SPEED_X
            
            # Add a bit of randomness to prevent loops
            self.ball_vel[0] += (self.np_random.random() - 0.5) * 0.2
            
            # Clamp speed
            self.ball_vel[0] = max(-self.BALL_MAX_SPEED_X, min(self.ball_vel[0], self.BALL_MAX_SPEED_X))
            
            # Reward for risky play
            if abs(offset) > 0.8:
                return 5.0 # Risky hit bonus
            else:
                return -2.0 # Safe hit penalty

        return 0.0

    def _check_block_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        for block in self.blocks[:]:
            if block["rect"].colliderect(ball_rect):
                # sound: break_block.wav
                
                # Determine collision side to correctly reflect the ball
                prev_ball_pos_x = self.ball_pos[0] - self.ball_vel[0]
                prev_ball_pos_y = self.ball_pos[1] - self.ball_vel[1]
                
                if (prev_ball_pos_x < block["rect"].left or prev_ball_pos_x > block["rect"].right):
                     self.ball_vel[0] *= -1
                else: # Top or bottom hit
                     self.ball_vel[1] *= -1

                self._create_particles(block["rect"].center, block["color"])
                reward = block["points"]
                self.score += reward
                self.blocks.remove(block)
                return reward, True
        
        return 0.0, False

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 5)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 30.0))))
            color = (*p["color"], alpha)
            size = max(1, int(3 * (p["lifetime"] / 30.0)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 80))
        self.screen.blit(glow_surf, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))
        # Ball itself
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Lives
        for i in range(self.lives):
            pos_x = self.WIDTH - 25 - (i * (self.BALL_RADIUS * 2 + 10))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            
        # Game Over / Win Message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            msg_text = self.font_msg.render(msg, True, color)
            text_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Create a semi-transparent background for the text
            bg_surf = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            bg_surf.fill((20, 20, 30, 180))
            self.screen.blit(bg_surf, (text_rect.left - 10, text_rect.top - 10))
            
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    env.validate_implementation()
    
    # Test a full episode
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = env.action_space.sample() # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
    print(f"Episode finished after {step_count} steps.")
    print(f"Final score: {info['score']}, Total reward: {total_reward:.2f}")
    
    env.close()