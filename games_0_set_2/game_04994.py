
# Generated: 2025-08-28T03:39:24.545113
# Source Brief: brief_04994.md
# Brief Index: 4994

        
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
        "Controls: ←→ to move the paddle. Break all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block-breaking game. Control the paddle to bounce the ball and destroy all the blocks on screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.PADDLE_WIDTH_INITIAL = 120
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 7
        self.MAX_BALL_SPEED_X = 9
        self.MAX_EPISODE_STEPS = 2000
        self.PADDLE_BOUNCE_FACTOR = 4

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (200, 200, 255, 50)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150, 100)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]
        self.COLOR_TEXT = (255, 255, 255)

        # Fonts
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_icons = pygame.font.SysFont("monospace", 32, bold=True)

        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.ball_trail = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.risky_hit_pending = False
        self.total_blocks = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.risky_hit_pending = False
        
        # Paddle
        self.paddle_rect = pygame.Rect(
            (self.screen_width - self.PADDLE_WIDTH_INITIAL) / 2,
            self.screen_height - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH_INITIAL,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        block_width = 60
        block_height = 20
        rows = 5
        cols = 10
        self.total_blocks = rows * cols
        for r in range(rows):
            for c in range(cols):
                block_rect = pygame.Rect(
                    c * (block_width + 4) + 2,
                    r * (block_height + 4) + 40,
                    block_width,
                    block_height
                )
                color_index = (r + c) % len(self.BLOCK_COLORS)
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index]})

        # Effects
        self.particles = []
        self.ball_trail = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = [
            self.BALL_SPEED_INITIAL * math.cos(angle),
            self.BALL_SPEED_INITIAL * math.sin(angle)
        ]
        self.ball_trail = []

    def step(self, action):
        reward = -0.01  # Time penalty
        terminated = False
        
        # 1. Handle player input
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.screen_width - self.paddle_rect.width, self.paddle_rect.x))

        # 2. Update game logic
        # Apply "safe play" penalty
        if self.ball_vel[1] > 0 and abs(self.paddle_rect.centerx - self.ball_pos[0]) < self.paddle_rect.width * 0.1:
            reward -= 0.2

        # Update ball trail
        self.ball_trail.append(list(self.ball_pos))
        if len(self.ball_trail) > 5:
            self.ball_trail.pop(0)

        # Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Ball collisions
        # Wall collision
        if ball_rect.left <= 0 or ball_rect.right >= self.screen_width:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.screen_width, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle_rect.top
            
            offset = ball_rect.centerx - self.paddle_rect.centerx
            normalized_offset = offset / (self.paddle_rect.width / 2)
            self.ball_vel[0] += normalized_offset * self.PADDLE_BOUNCE_FACTOR
            self.ball_vel[0] = max(-self.MAX_BALL_SPEED_X, min(self.MAX_BALL_SPEED_X, self.ball_vel[0]))
            
            # Check for risky hit
            if abs(normalized_offset) > 0.6: # Hit on outer 40%
                self.risky_hit_pending = True
            else:
                self.risky_hit_pending = False
            # sfx: paddle_hit

        # Block collision
        hit_block = None
        for i, block in enumerate(self.blocks):
            if ball_rect.colliderect(block["rect"]):
                hit_block = i
                break
        
        if hit_block is not None:
            block_data = self.blocks.pop(hit_block)
            block_rect = block_data["rect"]
            
            reward += 1.0 # Reward for hitting a block
            self.score += 10
            
            if self.risky_hit_pending:
                reward += 5.0
                self.risky_hit_pending = False
            
            # Create particles
            for _ in range(20):
                self._create_particle(block_rect.center, block_data["color"])

            # Bounce logic
            clip = ball_rect.clip(block_rect)
            if clip.width < clip.height:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1
            # sfx: block_break

        # Ball lost
        if ball_rect.top > self.screen_height:
            self.lives -= 1
            reward -= 50.0
            self.risky_hit_pending = False
            if self.lives > 0:
                self._reset_ball()
                # sfx: lose_life
            else:
                terminated = True
                # sfx: game_over

        # Win condition
        if not self.blocks:
            reward += 100.0
            terminated = True
            self.score += 1000 # Bonus for clearing
        
        # Update particles
        self._update_particles()
        
        # Step limit
        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particle(self, pos, color):
        particle = {
            "pos": list(pos),
            "vel": [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
            "lifespan": self.np_random.integers(15, 30),
            "color": color,
            "size": self.np_random.integers(3, 7)
        }
        self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["size"] = max(0, p["size"] - 0.2)
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            # Add a slight 3D effect
            brighter = tuple(min(255, c + 30) for c in block["color"])
            darker = tuple(max(0, c - 30) for c in block["color"])
            pygame.draw.line(self.screen, brighter, block["rect"].topleft, block["rect"].topright, 2)
            pygame.draw.line(self.screen, brighter, block["rect"].topleft, block["rect"].bottomleft, 2)
            pygame.draw.line(self.screen, darker, block["rect"].bottomright, block["rect"].topright, 2)
            pygame.draw.line(self.screen, darker, block["rect"].bottomright, block["rect"].bottomleft, 2)

        # Paddle glow and paddle
        glow_rect = self.paddle_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PADDLE_GLOW, s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)

        # Ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
            color = self.COLOR_BALL_GLOW[:3] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, color)

        # Ball glow and ball
        if self.ball_pos:
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p["color"], (p["pos"][0], p["pos"][1], int(p["size"]), int(p["size"])))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Lives
        lives_text = self.font_main.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.screen_width - 180, 5))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.screen_width - 80 + i * 25, 17, 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.screen_width - 80 + i * 25, 17, 8, self.COLOR_BALL)

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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
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
    obs, info = env.reset()
    
    # To demonstrate, let's also run it with a visible window
    del os.environ["SDL_VIDEODRIVER"]
    
    env_render = GameEnv(render_mode="rgb_array")
    obs, info = env_render.reset()
    
    # Setup for manual play
    screen = pygame.display.set_mode((env_render.screen_width, env_render.screen_height))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("MANUAL PLAY INSTRUCTIONS")
    print(env_render.user_guide)
    print("="*30 + "\n")

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
        
        if keys[pygame.K_ESCAPE]:
            running = False

        obs, reward, terminated, truncated, info = env_render.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env_render.reset()
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(30) # Run at 30 FPS
        
    env_render.close()