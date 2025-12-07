
# Generated: 2025-08-28T00:33:49.602250
# Source Brief: brief_03826.md
# Brief Index: 3826

        
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
        "Controls: ←→ to move the paddle. Break all the bricks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant arcade classic. Control the paddle to bounce the ball and destroy all the neon bricks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 5

        # Colors
        self.COLOR_BG_START = (10, 0, 30)
        self.COLOR_BG_END = (40, 0, 70)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (0, 255, 255)
        self.COLOR_BALL_GLOW = (0, 150, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = [
            (255, 0, 255), (255, 128, 0), (0, 255, 0), 
            (255, 255, 0), (0, 128, 255)
        ]

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_Y = self.SCREEN_HEIGHT - 30
        self.PADDLE_SPEED = 12

        # Ball settings
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED = 10.0
        self.BALL_SPEED_INCREMENT = 0.1

        # Brick settings
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 15
        self.TOTAL_BRICKS = self.BRICK_ROWS * self.BRICK_COLS
        self.BRICK_WIDTH = (self.SCREEN_WIDTH - (self.BRICK_COLS + 1) * 2) // self.BRICK_COLS
        self.BRICK_HEIGHT = 15
        self.BRICK_SPACING = 2
        self.BRICK_TOP_MARGIN = 50

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
        self.font = pygame.font.Font(None, 36)
        
        # Game state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.initial_brick_count = 0
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False

        # Paddle
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)  # Upwards
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.INITIAL_BALL_SPEED

        # Bricks
        self._generate_bricks()
        self.initial_brick_count = len(self.bricks)

        # Particles
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self._handle_movement(movement)
        self.ball_pos += self.ball_vel
        brick_reward = self._handle_collisions()
        reward += brick_reward
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Final reward on first termination frame
            if not self.bricks: # Win
                reward += 100
            elif self.lives <= 0: # Loss
                reward -= 100
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        return self.lives <= 0 or not self.bricks or self.steps >= self.MAX_STEPS

    def _generate_bricks(self):
        self.bricks = []
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = j * (self.BRICK_WIDTH + self.BRICK_SPACING) + self.BRICK_SPACING * 2
                brick_y = i * (self.BRICK_HEIGHT + self.BRICK_SPACING) + self.BRICK_TOP_MARGIN
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append({"rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT), "color": color})
    
    def _handle_movement(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

    def _handle_collisions(self):
        # Walls
        if self.ball_pos[0] - self.BALL_RADIUS <= 0 or self.ball_pos[0] + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce

        if self.ball_pos[1] - self.BALL_RADIUS <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Bottom of screen (lose life)
        if self.ball_pos[1] + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives > 0:
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                current_speed = np.linalg.norm(self.ball_vel)
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * current_speed
                # sfx: lose_life
            return 0

        # Paddle
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            
            current_speed = np.linalg.norm(self.ball_vel)
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * current_speed
            # sfx: paddle_bounce

        # Bricks
        bricks_destroyed = 0
        for i in range(len(self.bricks) - 1, -1, -1):
            brick = self.bricks[i]
            if brick["rect"].colliderect(ball_rect):
                prev_ball_pos = self.ball_pos - self.ball_vel
                if (prev_ball_pos[1] + self.BALL_RADIUS <= brick["rect"].top or
                    prev_ball_pos[1] - self.BALL_RADIUS >= brick["rect"].bottom):
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1

                self._create_particles(brick["rect"].center, brick["color"])
                self.score += 10
                bricks_destroyed += 1
                del self.bricks[i]
                # sfx: brick_destroy
                break

        if bricks_destroyed > 0:
            bricks_destroyed_total = self.initial_brick_count - len(self.bricks)
            speed_increases = bricks_destroyed_total // 10
            new_speed = min(self.INITIAL_BALL_SPEED + speed_increases * self.BALL_SPEED_INCREMENT, self.MAX_BALL_SPEED)
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * new_speed

        return bricks_destroyed * 1.0

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
    
    def _get_observation(self):
        # Clear screen with background
        self._draw_gradient_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
    
    def _render_game(self):
        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            highlight_color = tuple(min(255, c + 50) for c in brick["color"])
            pygame.draw.rect(self.screen, highlight_color, brick["rect"].inflate(-4, -4))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30))))
            color = p["color"]
            pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), 2)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            end_text_str = "YOU WIN!" if not self.bricks else "GAME OVER"
            end_text_color = self.COLOR_BALL if not self.bricks else (255, 0, 100)
            end_text = self.font.render(end_text_str, True, end_text_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to run the file directly to see the game
    # It will open a pygame window and let you play.
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run validation in headless mode ---
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # env_headless = GameEnv()
    # env_headless.validate_implementation()
    # env_headless.close()
    # print("\n--- Starting interactive mode ---")
    
    # --- Interactive mode ---
    obs, info = env.reset()
    terminated = False
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Brick Breaker")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if terminated:
            # Wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                terminated = False
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()