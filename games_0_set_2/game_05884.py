
# Generated: 2025-08-28T06:22:34.601409
# Source Brief: brief_05884.md
# Brief Index: 5884

        
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
        "A retro block-breaking game. Use the paddle to keep the ball in play and destroy all the blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (100, 100, 120)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]

        # Paddle properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12

        # Ball properties
        self.BALL_RADIUS = 7
        self.BALL_BASE_SPEED = 6.0
        self.BALL_MAX_HORIZONTAL_SPEED = 5.5

        # Block properties
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.BLOCK_WIDTH = (self.WIDTH - 40) // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 20
        self.BLOCK_SPACING = 4
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_state = None
        self.blocks = None
        self.lives = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_blocks = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.particles = []
        self._generate_blocks()
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_state = "held"
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        start_x = (self.WIDTH - (self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING)) / 2
        start_y = 50
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT), "color": color})
        self.total_blocks = len(self.blocks)

    def step(self, action):
        reward = -0.01  # Time penalty to encourage faster completion

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        if not self.game_over:
            # Handle paddle movement
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

            # Handle ball launch
            if self.ball_state == "held" and space_held:
                # sfx: ball_launch
                self.ball_state = "in_play"
                initial_angle = (random.random() - 0.5) * (math.pi / 4) # -22.5 to +22.5 degrees
                self.ball_vel = [
                    self.BALL_BASE_SPEED * math.sin(initial_angle),
                    -self.BALL_BASE_SPEED * math.cos(initial_angle)
                ]

            # Update game state
            reward += self._update_ball()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if len(self.blocks) == 0: # Win condition
                reward += 100
                self.game_over = True
            elif self.lives <= 0: # Lose condition
                reward -= 100
                self.game_over = True
            elif self.steps >= self.MAX_STEPS: # Max steps reached
                self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self):
        if self.ball_state == "held":
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            return 0

        # Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        ball_rect.center = (int(self.ball_pos[0]), int(self.ball_pos[1]))

        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_pos[0] = ball_rect.centerx
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.right >= self.WIDTH:
            ball_rect.right = self.WIDTH
            self.ball_pos[0] = ball_rect.centerx
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery
            
            offset = ball_rect.centerx - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            
            self.ball_vel[0] = self.BALL_MAX_HORIZONTAL_SPEED * normalized_offset
            
            speed_sq = self.BALL_BASE_SPEED ** 2
            vy_sq = speed_sq - self.ball_vel[0]**2
            
            if vy_sq < 0: # Should not happen if max horizontal speed is less than base speed
                self.ball_vel[0] = np.sign(self.ball_vel[0]) * self.BALL_MAX_HORIZONTAL_SPEED
                vy_sq = speed_sq - self.ball_vel[0]**2

            self.ball_vel[1] = -math.sqrt(vy_sq)

        # Block collisions
        step_reward = 0
        for block_data in self.blocks[:]:
            block_rect = block_data["rect"]
            if ball_rect.colliderect(block_rect):
                # sfx: block_break
                self._create_particles(block_rect.center, block_data["color"])
                self.blocks.remove(block_data)
                self.score += 1
                step_reward += 1.0

                # Collision resolution
                dx = ball_rect.centerx - block_rect.centerx
                dy = ball_rect.centery - block_rect.centery
                
                pen_x = (ball_rect.width / 2 + block_rect.width / 2) - abs(dx)
                pen_y = (ball_rect.height / 2 + block_rect.height / 2) - abs(dy)

                if pen_x < pen_y: # Horizontal collision
                    self.ball_vel[0] *= -1
                    self.ball_pos[0] += np.sign(self.ball_vel[0]) * pen_x
                else: # Vertical collision
                    self.ball_vel[1] *= -1
                    self.ball_pos[1] += np.sign(self.ball_vel[1]) * pen_y
                break # Only handle one block collision per frame

        # Ball lost
        if ball_rect.top > self.HEIGHT:
            # sfx: lose_life
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return step_reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 25),
                "radius": random.uniform(2, 5),
                "color": color
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return len(self.blocks) == 0 or self.lives <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                alpha = max(0, min(255, int(255 * (p["life"] / 25))))
                color = (*p["color"], alpha)
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_PADDLE)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"BALLS: {self.lives}", True, self.COLOR_PADDLE)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if len(self.blocks) == 0:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv(render_mode="rgb_array")
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode terminated.")
            obs, info = env.reset()
            print("Environment reset.")

    # A simple interactive loop (requires a display)
    try:
        del os.environ["SDL_VIDEODRIVER"]
        env.close() # Close the headless env
        
        interactive_env = GameEnv(render_mode="rgb_array")
        obs, info = interactive_env.reset()
        
        screen = pygame.display.set_mode((interactive_env.WIDTH, interactive_env.HEIGHT))
        pygame.display.set_caption("Block Breaker Gym Environment")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # no-op
            space = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1

            action = [movement, space, 0] # shift is unused
            
            obs, reward, terminated, truncated, info = interactive_env.step(action)
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000) # Pause for 2 seconds
                obs, info = interactive_env.reset()

            clock.tick(30) # Run at 30 FPS

        interactive_env.close()

    except pygame.error as e:
        print("\nCould not start interactive mode. Pygame display not available.")
        print("This is expected if you are in a headless environment (like a server or notebook).")
    except Exception as e:
        print(f"An error occurred during interactive test: {e}")
    finally:
        env.close()