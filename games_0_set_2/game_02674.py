
# Generated: 2025-08-27T21:05:42.310779
# Source Brief: brief_02674.md
# Brief Index: 2674

        
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
        "A fast-paced, top-down block breaker. Break all the blocks to win. Risky paddle hits at steep angles are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        
        # --- Game Constants ---
        self.MAX_STEPS = 2500
        self.INITIAL_BALLS = 3

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 230, 0)
        self.COLOR_BALL_GLOW = (255, 180, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (255, 60, 60), (60, 255, 60), (60, 60, 255),
            (255, 255, 60), (60, 255, 255), (255, 60, 255)
        ]

        # Paddle
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_Y = self.HEIGHT - 40
        self.PADDLE_SPEED = 12

        # Ball
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 6
        self.BALL_MAX_X_VEL = self.BALL_SPEED * 1.2

        # Blocks
        self.NUM_BLOCKS_X = 10
        self.NUM_BLOCKS_Y = 6
        self.TOTAL_BLOCKS = self.NUM_BLOCKS_X * self.NUM_BLOCKS_Y
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 18
        self.BLOCK_SPACING = 6
        self.BLOCK_AREA_Y_START = 50

        # UI
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.blocks_left = 0
        self.combo = 0
        self.game_over = False
        
        # Initialize state
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = self.INITIAL_BALLS
        self.blocks_left = self.TOTAL_BLOCKS
        self.game_over = False
        self.combo = 1
        
        # Paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        for j in range(self.NUM_BLOCKS_Y):
            for i in range(self.NUM_BLOCKS_X):
                x = i * (self.BLOCK_WIDTH + self.BLOCK_SPACING) + self.BLOCK_SPACING * 2
                y = j * (self.BLOCK_HEIGHT + self.BLOCK_SPACING) + self.BLOCK_AREA_Y_START
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({
                    "rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    "color": color,
                    "active": True
                })

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty to encourage speed
        
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Update paddle position
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # 2. Launch ball
        if space_held and not self.ball_launched:
            # sfx: launch_ball.wav
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED

        # 3. Update ball position and collisions
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx
        else:
            self.ball_pos += self.ball_vel

            # Wall collisions
            if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
                self.ball_vel.x *= -1
                self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce.wav
            if self.ball_pos.y - self.BALL_RADIUS <= 0:
                self.ball_vel.y *= -1
                self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce.wav

            # Floor collision (lose ball)
            if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
                self.balls_left -= 1
                self.combo = 1
                # sfx: lose_ball.wav
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    reward -= 100

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
                # sfx: paddle_hit.wav
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                self.ball_vel.y *= -1
                
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                offset = np.clip(offset, -1, 1)
                self.ball_vel.x = self.BALL_MAX_X_VEL * offset
                
                # Risk/reward for hit angle
                if abs(offset) > 0.7:
                    reward += 0.1 # Risky hit
                    self.combo += 1
                else:
                    reward -= 0.2 # Cautious hit
                    self.combo = 1 # Reset combo on safe hits
            
            # Block collisions
            for block in self.blocks:
                if block["active"] and block["rect"].colliderect(ball_rect):
                    # sfx: block_break.wav
                    block["active"] = False
                    self.blocks_left -= 1
                    
                    block_reward = 1 * self.combo
                    reward += block_reward
                    self.score += block_reward
                    
                    self._create_particles(block["rect"].center, block["color"])

                    # Determine bounce direction
                    # Check if collision is more horizontal or vertical
                    dx = abs(self.ball_pos.x - block["rect"].centerx)
                    dy = abs(self.ball_pos.y - block["rect"].centery)
                    
                    if dx / block["rect"].width > dy / block["rect"].height:
                        self.ball_vel.x *= -1
                    else:
                        self.ball_vel.y *= -1
                    
                    break # Only break one block per frame
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Update step counter
        self.steps += 1
        
        # 6. Check termination conditions
        terminated = self.game_over
        if self.blocks_left == 0:
            terminated = True
            reward += 100 # Win bonus
        if self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Blocks
        for block in self.blocks:
            if block["active"]:
                pygame.draw.rect(self.screen, block["color"], block["rect"])
                # Add a slight 3D effect
                brighter = tuple(min(255, c + 30) for c in block["color"])
                darker = tuple(max(0, c - 30) for c in block["color"])
                pygame.draw.line(self.screen, brighter, block["rect"].topleft, block["rect"].topright, 2)
                pygame.draw.line(self.screen, brighter, block["rect"].topleft, block["rect"].bottomleft, 2)
                pygame.draw.line(self.screen, darker, block["rect"].bottomright, block["rect"].topright, 2)
                pygame.draw.line(self.screen, darker, block["rect"].bottomright, block["rect"].bottomleft, 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"].x), int(p["pos"].y)))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), glow_radius, (*self.COLOR_BALL_GLOW, 80))
        # Main ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_PADDLE)

        # Combo
        if self.combo > 1:
            combo_colors = [(0, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0)]
            color_index = min(self.combo - 2, len(combo_colors) - 1)
            combo_color = combo_colors[color_index]
            combo_text = f"x{self.combo} COMBO"
            combo_surf = self.font_medium.render(combo_text, True, combo_color)
            pos_x = self.paddle.centerx - combo_surf.get_width() / 2
            pos_y = self.paddle.y - 35
            self.screen.blit(combo_surf, (int(pos_x), int(pos_y)))
        
        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.blocks_left == 0:
                text = "YOU WIN!"
                color = (100, 255, 100)
            else:
                text = "GAME OVER"
                color = (255, 100, 100)

            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": self.blocks_left,
            "combo": self.combo,
        }

    def close(self):
        pygame.font.quit()
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
        
        # Test assertions from brief
        assert 0 <= self.blocks_left <= self.TOTAL_BLOCKS
        assert 0 <= self.balls_left <= self.INITIAL_BALLS
        if self.ball_launched:
            assert self.ball_vel.magnitude() < self.BALL_MAX_X_VEL * 2
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
            
        # The action is a list/tuple
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose the obs from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()