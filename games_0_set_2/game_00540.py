
# Generated: 2025-08-27T13:57:28.112139
# Source Brief: brief_00540.md
# Brief Index: 540

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down block breaker. Control the paddle to bounce the ball, destroy all the blocks, and get a high score."
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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (200, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            10: (0, 255, 100),   # Green
            20: (100, 100, 255), # Blue
            30: (255, 50, 50)    # Red
        }

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7
        self.MAX_STEPS = 2000
        self.WALL_THICKNESS = 10
        
        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        paddle_y = self.HEIGHT - 40
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._generate_blocks()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 5], dtype=float)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upward angle
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.BALL_SPEED

    def _generate_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        rows = 5
        cols = 10
        top_margin = 60
        x_gap = 6
        y_gap = 6
        
        points_map = [30, 30, 20, 20, 10] # Points per row from top to bottom

        for r in range(rows):
            for c in range(cols):
                points = points_map[r]
                color = self.BLOCK_COLORS[points]
                block_rect = pygame.Rect(
                    self.WALL_THICKNESS + c * (block_width + x_gap),
                    top_margin + r * (block_height + y_gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "color": color, "points": points})

    def step(self, action):
        reward = -0.01  # Time penalty to encourage efficiency
        
        # 1. Handle player input
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(self.WALL_THICKNESS, min(self.paddle_rect.x, self.WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # 2. Update ball position
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # 3. Handle collisions
        # Wall collisions
        if self.ball_pos[0] <= self.WALL_THICKNESS + self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.WALL_THICKNESS + self.BALL_RADIUS, self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.WALL_THICKNESS + self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.WALL_THICKNESS + self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
            
            # Add reward for hitting the ball
            reward += 0.1

            # Calculate horizontal influence and risk/reward
            dist_from_center = self.ball_pos[0] - self.paddle_rect.centerx
            normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
            
            if abs(normalized_dist) < 0.3: # Safe bounce near center
                reward -= 20
            else: # Risky bounce near edge
                reward += 40
            
            self.ball_vel[0] += normalized_dist * 2.0  # Apply horizontal "english"
            
            # Normalize speed to prevent acceleration
            speed = np.linalg.norm(self.ball_vel)
            if speed > 0:
                self.ball_vel = (self.ball_vel / speed) * self.BALL_SPEED
            
            # Add slight randomness to vertical velocity to prevent loops
            self.ball_vel[1] += self.np_random.uniform(-0.1, 0.1)
            # sfx: paddle_hit

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block["rect"]):
                hit_block = block
                break
        
        if hit_block:
            self.blocks.remove(hit_block)
            reward += hit_block["points"]
            self.score += hit_block["points"]
            self.ball_vel[1] *= -1 # Simple vertical bounce
            self._create_particles(hit_block["rect"].center, hit_block["color"])
            # sfx: block_break

        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination conditions
        terminated = False
        
        # Lost ball
        if self.ball_pos[1] > self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life
            if self.lives <= 0:
                terminated = True
                reward -= 100
            else:
                self._reset_ball()

        # Win condition
        if not self.blocks:
            terminated = True
            reward += 100
            # sfx: win_game
            
        # Max steps
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
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
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "radius": radius, "color": color, "life": lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)

        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p["radius"])), p["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], max(0, int(p["radius"])), p["color"])

        # Draw ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 10))
        
        # Lives
        lives_text = self.font.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 10, self.WALL_THICKNESS + 10))

        if self.game_over:
            msg = "GAME OVER"
            if not self.blocks:
                msg = "YOU WIN!"
            
            end_text = self.font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    print(env.game_description)
    print(env.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # 0=none

        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over. Final Info: {info}")
    env.close()