
# Generated: 2025-08-28T00:04:32.055548
# Source Brief: brief_03680.md
# Brief Index: 3680

        
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
        "A fast-paced, top-down block breaker. Clear all the blocks before you run out of balls."
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
        
        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 6.0
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 20, 45)
        self.COLOR_BG_GRID = (25, 30, 55)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 70, 70), (255, 140, 70), (255, 210, 70),
            (70, 255, 70), (70, 210, 255), (140, 70, 255)
        ]

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.game_over = False
        self.chain_count = 0
        
        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.game_over = False
        self.chain_count = 0
        self.particles = []

        # Paddle
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self._reset_ball()
        
        # Blocks
        self.blocks = []
        self.block_colors = {}
        block_rows = 6
        block_cols = 10
        block_width = 58
        block_height = 20
        total_block_width = block_cols * (block_width + 4) - 4
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50
        for i in range(block_rows):
            for j in range(block_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block = pygame.Rect(
                    start_x + j * (block_width + 4),
                    start_y + i * (block_height + 4),
                    block_width,
                    block_height
                )
                self.blocks.append(block)
                self.block_colors[id(block)] = color
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball = {
            "pos": [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS],
            "vel": [0, 0],
            "held": True,
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Initialize step reward
        reward = -0.02

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        if self.ball["held"] and space_held:
            # Sound: launch.wav
            self.ball["held"] = False
            initial_angle = self.np_random.uniform(-math.pi/4, math.pi/4)
            self.ball["vel"] = [
                self.BALL_SPEED_INITIAL * math.sin(initial_angle),
                -self.BALL_SPEED_INITIAL * math.cos(initial_angle)
            ]

        # 2. Update game logic
        self._update_ball()
        self._update_particles()
        
        # 3. Handle collisions and calculate rewards
        reward += self._handle_collisions()

        # 4. Update step counter and check for termination
        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over:
             # Check for win condition
            if not self.blocks:
                reward += 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        if self.ball["held"]:
            self.ball["pos"][0] = self.paddle.centerx
            self.ball["pos"][1] = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball["pos"][0] += self.ball["vel"][0]
            self.ball["pos"][1] += self.ball["vel"][1]

    def _handle_collisions(self):
        reward = 0
        ball_pos = self.ball["pos"]
        ball_vel = self.ball["vel"]
        ball_rect = pygame.Rect(
            ball_pos[0] - self.BALL_RADIUS,
            ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        # Wall collisions
        if ball_pos[0] <= self.BALL_RADIUS:
            ball_pos[0] = self.BALL_RADIUS
            ball_vel[0] *= -1
        if ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            ball_vel[0] *= -1
        if ball_pos[1] <= self.BALL_RADIUS:
            ball_pos[1] = self.BALL_RADIUS
            ball_vel[1] *= -1
        
        # Bottom wall (lose ball)
        if ball_pos[1] >= self.HEIGHT:
            # Sound: lose_ball.wav
            self.balls_left -= 1
            reward -= 10
            self.chain_count = 0
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward

        # Paddle collision
        if not self.ball["held"] and ball_rect.colliderect(self.paddle):
            if ball_vel[1] > 0: # Only collide if moving downwards
                # Sound: paddle_hit.wav
                self.ball["pos"][1] = self.paddle.top - self.BALL_RADIUS
                offset = (ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                angle = offset * (math.pi / 2.5) # Max angle ~72 degrees
                
                speed = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2)
                ball_vel[0] = speed * math.sin(angle)
                ball_vel[1] = -speed * math.cos(angle)
                self.chain_count = 0

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block):
                # Sound: block_break.wav
                self._create_particles(block.center, self.block_colors[id(block)])
                self.blocks.remove(block)
                reward += 1
                self.chain_count += 1
                if self.chain_count >= 3:
                    reward += 5 # Chain bonus
                
                # Determine collision side to correctly reflect the ball
                prev_ball_pos = [ball_pos[0] - ball_vel[0], ball_pos[1] - ball_vel[1]]
                prev_ball_rect = pygame.Rect(
                    prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS,
                    self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
                )
                
                # Simple but effective collision response
                if prev_ball_rect.bottom <= block.top or prev_ball_rect.top >= block.bottom:
                    ball_vel[1] *= -1
                else:
                    ball_vel[0] *= -1
                
                break # Handle one collision per frame

        return reward

    def _check_termination(self):
        return self.balls_left <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
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
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (0, i), (self.WIDTH, i))

        # Blocks
        for block in self.blocks:
            color = self.block_colors[id(block)]
            pygame.draw.rect(self.screen, color, block, border_radius=3)
            # Add a slight 3D effect
            highlight = tuple(min(255, c + 30) for c in color)
            shadow = tuple(max(0, c - 30) for c in color)
            pygame.draw.line(self.screen, highlight, block.topleft, block.topright, 2)
            pygame.draw.line(self.screen, highlight, block.topleft, block.bottomleft, 2)
            pygame.draw.line(self.screen, shadow, block.bottomright, block.topright, 2)
            pygame.draw.line(self.screen, shadow, block.bottomright, block.bottomleft, 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"]
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (1, 1), 1)
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

        # Paddle
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE)
        
        # Ball
        x, y = int(self.ball["pos"][0]), int(self.ball["pos"][1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        # Add a little glow
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS + 2, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS + 4, (*self.COLOR_BALL, 30))

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Balls left
        ball_text_surf = self.font_main.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text_surf, (self.WIDTH - 180, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 70 + i * 25, 27, self.BALL_RADIUS - 2, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 70 + i * 25, 27, self.BALL_RADIUS - 2, self.COLOR_BALL)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "GAME OVER" if self.balls_left <= 0 else "YOU WIN!"
            text_surf = self.font_main.render(message, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, (220, 220, 220))
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)

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
    # Set this to 'human' to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # In human mode, we need a real screen
        pygame.display.set_caption("Block Breaker")
        real_screen = pygame.display.set_mode((640, 400))
    
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # Simple agent: move left and right randomly, launch ball when possible
    while not terminated:
        if render_mode == "human":
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            # Simple keyboard control for human play
            keys = pygame.key.get_pressed()
            movement_action = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement_action, space_action, shift_action]
        
        else: # Random agent for rgb_array mode
            action = env.action_space.sample()
            # Make random agent more active
            if random.random() < 0.8:
                action[0] = random.choice([3, 4]) # Move left/right
            if random.random() < 0.1:
                action[1] = 1 # Press space
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # Blit the env's screen to the real screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit to 30 FPS for human play
            
    print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
    env.close()