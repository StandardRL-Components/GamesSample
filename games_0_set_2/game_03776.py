
# Generated: 2025-08-28T00:23:22.274235
# Source Brief: brief_03776.md
# Brief Index: 3776

        
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
    user_guide = "Controls: ←→ to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down block breaker where you strategically deflect a ball to destroy all blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (35, 35, 45)
        self.COLOR_PADDLE = (200, 200, 200)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 87, 34), (255, 193, 7), (76, 175, 80), 
            (33, 150, 243), (156, 39, 176), (233, 30, 99)
        ]

        # Game parameters
        self.MAX_STEPS = 1000
        self.INITIAL_BALLS = 3
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 4.0
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.balls_left = 0
        self.game_over = False
        self.game_won = False
        self.total_blocks = 0
        
        # Initialize state variables
        self.reset()
    
    def _create_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 5
        cols = 10
        gap = 4
        
        grid_width = cols * (block_width + gap) - gap
        start_x = (self.WIDTH - grid_width) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({"rect": block_rect, "color": color})
        self.total_blocks = len(self.blocks)

    def _launch_ball(self):
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )
        angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
        self.ball_vel = [
            self.BALL_SPEED_INITIAL * math.cos(angle),
            self.BALL_SPEED_INITIAL * math.sin(angle),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._create_blocks()
        self._launch_ball()
        
        self.particles = []
        self.steps = 0
        self.score = 0
        self.balls_left = self.INITIAL_BALLS
        self.game_over = False
        self.game_won = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Unused
        # shift_held = action[2] == 1  # Unused

        reward = 0
        self.steps += 1

        if movement == 3:  # Move left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02
        elif movement == 4:  # Move right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if not self.game_over:
            # Update ball position
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

            # Ball collision with walls
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.x = max(0, min(self.WIDTH - self.ball.width, self.ball.x))
                # sfx: wall_bounce
            if self.ball.top <= 0:
                self.ball_vel[1] *= -1
                self.ball.y = max(0, min(self.HEIGHT - self.ball.height, self.ball.y))
                # sfx: wall_bounce

            # Ball collision with paddle
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # sfx: paddle_hit
                self.ball.bottom = self.paddle.top
                self.ball_vel[1] *= -1
                
                hit_pos_norm = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += hit_pos_norm * 2.0
                
                speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED_INITIAL
                    self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED_INITIAL

            # Ball collision with blocks
            hit_block_idx = self.ball.collidelist([b["rect"] for b in self.blocks])
            if hit_block_idx != -1:
                # sfx: block_break
                block_info = self.blocks.pop(hit_block_idx)
                block_rect = block_info["rect"]
                
                for _ in range(15):
                    self._spawn_particle(block_rect.center, block_info["color"])

                # Determine bounce direction
                dx = self.ball.centerx - block_rect.centerx
                dy = self.ball.centery - block_rect.centery
                w = (self.ball.width + block_rect.width) / 2
                h = (self.ball.height + block_rect.height) / 2
                wy = w * dy
                hx = h * dx

                if wy > hx:
                    if wy > -hx: # Top
                        self.ball_vel[1] *= -1; self.ball.bottom = block_rect.top
                    else: # Left
                        self.ball_vel[0] *= -1; self.ball.right = block_rect.left
                else:
                    if wy > -hx: # Right
                        self.ball_vel[0] *= -1; self.ball.left = block_rect.right
                    else: # Bottom
                        self.ball_vel[1] *= -1; self.ball.top = block_rect.bottom
                
                reward += 1
                self.score += 10
                
                if not self.blocks:
                    self.game_won = True; self.game_over = True; reward += 100

            # Ball loss
            if self.ball.top > self.HEIGHT:
                # sfx: lose_ball
                self.balls_left -= 1
                reward -= 5
                if self.balls_left > 0:
                    self._launch_ball()
                else:
                    self.game_over = True; reward -= 100

        self._update_particles()
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_particle(self, pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        lifespan = self.np_random.integers(15, 30)
        self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]; p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95; p["vel"][1] *= 0.95
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        
        # Render all game elements
        self._draw_particles()
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)
        
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"]
            size = int(max(1, self.BALL_RADIUS * 0.5 * (p["lifespan"] / 30)))
            surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, alpha), (size, size), size)
            self.screen.blit(surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.balls_left):
            pos = (self.WIDTH - 20 - i * (self.BALL_RADIUS * 2 + 5), 10 + self.BALL_RADIUS)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_game_over(self):
        text_str = "YOU WON!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        text_surf = self.font_game_over.render(text_str, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "won": self.game_won
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

# Example of how to run the environment for testing
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless for validation
    
    env = GameEnv()
    env.validate_implementation()
    
    # Test a full episode with random actions
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    print(f"Episode finished after {step_count} steps.")
    print(f"Final score: {info['score']}, Total reward: {total_reward:.2f}")
    print(f"Game won: {info['won']}")
    env.close()

    # To visualize the game, comment out the "dummy" driver line and
    # run the following interactive loop.
    #
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    #
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Block Breaker")
    # clock = pygame.time.Clock()
    #
    # obs, info = env.reset()
    # done = False
    #
    # while not done:
    #     movement = 0 # No-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         movement = 3
    #     elif keys[pygame.K_RIGHT]:
    #         movement = 4
    #
    #     action = [movement, 0, 0] # space and shift are unused
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #
    #     # Render the observation to the display screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     clock.tick(60) # Limit frame rate for human play
    #
    # env.close()