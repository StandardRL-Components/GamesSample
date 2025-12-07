
# Generated: 2025-08-27T21:35:02.565129
# Source Brief: brief_02838.md
# Brief Index: 2838

        
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
    A fast-paced top-down block breaker where strategic paddle positioning and risky plays are rewarded.
    The player controls a paddle to deflect a ball, breaking a grid of blocks.
    The goal is to clear all blocks without losing all three lives.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade block breaker. Clear all blocks by deflecting the ball. "
        "Risky deflections off the paddle's edge earn bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8.0
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED_X = 5.0
        self.BLOCK_COLS, self.BLOCK_ROWS = 10, 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 15
        self.BLOCK_SPACING = 6
        self.BLOCK_GRID_Y_OFFSET = 50
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        
        # --- Color Palette (Retro Arcade) ---
        self.COLOR_BG = (10, 10, 30)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BOUNDARY = (200, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            1: (0, 200, 100),   # Green
            2: (100, 150, 255), # Blue
            5: (255, 100, 100)  # Red
        }

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Internal State Variables ---
        # These are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.blocks_destroyed_count = None
        self.speed_increase_milestones = None
        self.last_ball_pos = None
        self.stuck_counter = None

        # Initialize state variables
        self.reset()
        
        # Validate implementation after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_launched = False
        self._reset_ball()

        self.blocks = self._create_blocks()
        self.total_blocks = len(self.blocks)
        
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.blocks_destroyed_count = 0
        self.speed_increase_milestones = {20, 40, 60, 80}
        self.last_ball_pos = [0, 0]
        self.stuck_counter = 0

        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - grid_width) / 2
        
        for row in range(self.BLOCK_ROWS):
            for col in range(self.BLOCK_COLS):
                x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_GRID_Y_OFFSET + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                
                # Determine block type/value
                if row < 2:
                    value = 5  # Red blocks
                elif row < 6:
                    value = 2  # Blue blocks
                else:
                    value = 1  # Green blocks
                
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                blocks.append({"rect": block_rect, "value": value})
        return blocks
    
    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.metadata["render_fps"])

        reward = -0.01  # Small penalty for each step to encourage efficiency
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle Actions ---
        movement = action[0]
        space_held = action[1] == 1

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Ball Launch
        if not self.ball_launched and space_held:
            self.ball_launched = True
            angle = (self.np_random.random() * 0.4 - 0.2) * math.pi # -36 to 36 degrees
            self.ball_vel = [
                self.INITIAL_BALL_SPEED * math.sin(angle),
                -self.INITIAL_BALL_SPEED * math.cos(angle)
            ]
            # sfx: ball_launch

        # --- Update Game Logic ---
        if self.ball_launched:
            reward += self._handle_ball_movement_and_collisions()
        else:
            # Keep ball attached to paddle
            self.ball_pos[0] = self.paddle.centerx

        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100 # Penalty for losing
        elif self.blocks_destroyed_count == self.total_blocks:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100 # Bonus for winning
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_ball_movement_and_collisions(self):
        step_reward = 0
        
        # Soft-lock detection
        if self.ball_pos == self.last_ball_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_ball_pos = list(self.ball_pos)
        
        if self.stuck_counter > 50:
            self.ball_vel[1] += (self.np_random.random() - 0.5) * 2
            self.stuck_counter = 0

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.MAX_BALL_SPEED_X * offset
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            
            # Risk/reward for paddle hits
            if abs(offset) > 0.9:
                step_reward += 2.0 # Risky hit bonus
            else:
                step_reward -= 0.2 # Safe hit penalty
            # sfx: paddle_hit
        
        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                step_reward += block["value"]
                self.score += block["value"]
                self._create_particles(block["rect"].center, self.BLOCK_COLORS[block["value"]])
                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Reverse ball direction (simple bounce)
                self.ball_vel[1] *= -1
                # sfx: block_destroy

                # Check for speed increase
                if self.blocks_destroyed_count in self.speed_increase_milestones:
                    self.speed_increase_milestones.remove(self.blocks_destroyed_count)
                    speed_magnitude = math.hypot(*self.ball_vel)
                    if speed_magnitude > 0:
                        scale = (speed_magnitude + 0.5) / speed_magnitude
                        self.ball_vel[0] *= scale
                        self.ball_vel[1] *= scale
                break

        # Out of bounds
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            self._reset_ball()
            # sfx: life_lost

        return step_reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
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
        # Draw play area boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Draw blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[block["value"]]
            pygame.draw.rect(self.screen, color, block["rect"])
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 30.0))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with antialiasing
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Render score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))
        
        # Render lives
        lives_text = self.font_medium.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (15, 10))
        heart_surf = self.font_medium.render("♥", True, (255, 80, 80))
        for i in range(self.lives):
            self.screen.blit(heart_surf, (lives_text.get_width() + 25 + i * 25, 10))

        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- For interactive play ---
    # This loop is for human interaction and visualization, not for training
    # It requires a window to be created, which is not the standard for gym envs.
    
    try:
        import os
        # Set a non-dummy driver if you want to see the window
        if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] == "dummy":
            print("NOTE: To see the game window, run with a display available.")
            print("      (e.g., on Linux, run `export SDL_VIDEODRIVER=x11` before the script)")
            raise ImportError("No display available for interactive mode.")
            
        window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + "="*30)
        print(f"GAME: {env.game_description}")
        print(f"CONTROLS: {env.user_guide}")
        print("="*30 + "\n")
        
        while not terminated:
            # --- Action Mapping for Human Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering to Screen ---
            # The observation is already a rendered frame, we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            window.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling (for quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

        print(f"Game Over! Final Score: {info['score']}")
        
    except (ImportError, pygame.error) as e:
        print(f"Skipping interactive example: {e}")
        # Test the environment API without rendering a window
        print("\nRunning a short headless test...")
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print("Episode finished.")
                obs, info = env.reset()
        print("Headless test completed.")

    finally:
        env.close()