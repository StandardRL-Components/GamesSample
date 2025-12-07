
# Generated: 2025-08-28T01:24:49.751122
# Source Brief: brief_04100.md
# Brief Index: 4100

        
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

    user_guide = (
        "Controls: Use ← and → to move the paddle."
    )

    game_description = (
        "A retro arcade game where you control a paddle to bounce a ball and break all the blocks. "
        "Hitting the ball with the edge of the paddle is risky but yields more points."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    BALL_RADIUS = 8
    WALL_THICKNESS = 10
    PADDLE_SPEED = 12
    BALL_BASE_SPEED = 7
    MAX_STEPS = 2000
    INITIAL_LIVES = 3
    NUM_BLOCK_ROWS = 4
    NUM_BLOCK_COLS = 5
    BLOCK_WIDTH = (WIDTH - 2 * WALL_THICKNESS) / NUM_BLOCK_COLS
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 4
    BLOCK_START_Y = 50

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 100)
    COLOR_WALL = (100, 100, 120)
    COLOR_TEXT = (255, 255, 255)
    COLOR_HEART = (255, 50, 50)
    BLOCK_COLORS = [(100, 255, 100), (100, 100, 255), (255, 100, 100), (255, 255, 100)]


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_hit_type = 'neutral' # 'safe', 'risky', 'neutral'

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.last_hit_type = 'neutral'
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._reset_ball()
        self._initialize_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage efficiency

        self._handle_input(action)
        self._update_ball_position()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._update_particles()
        
        if self.ball_pos.y > self.HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            # SFX: Lose life
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if not self.blocks: # Victory
                reward += 100
            elif self.lives <= 0: # Failure
                reward -= 100
        
        self.score += collision_reward # Only update score on events, not step penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = max(
            self.WALL_THICKNESS,
            min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS)
        )

    def _update_ball_position(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos.x - self.BALL_RADIUS,
            self.ball_pos.y - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collisions
        if self.ball_pos.x <= self.WALL_THICKNESS + self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.ball_pos.x, self.WALL_THICKNESS + self.BALL_RADIUS + 1)
            self.ball_pos.x = min(self.ball_pos.x, self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS - 1)
            # SFX: Wall bounce

        if self.ball_pos.y <= self.WALL_THICKNESS + self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.WALL_THICKNESS + self.BALL_RADIUS + 1
            # SFX: Wall bounce

        # Paddle collision
        if self.ball_vel.y > 0 and self.paddle.colliderect(ball_rect):
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS - 1

            hit_offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = self.BALL_BASE_SPEED * hit_offset * 1.5
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_BASE_SPEED

            # Determine hit type for reward on next block break
            if abs(hit_offset) < 0.2:
                self.last_hit_type = 'safe'
            elif abs(hit_offset) > 0.7:
                self.last_hit_type = 'risky'
            else:
                self.last_hit_type = 'neutral'
            # SFX: Paddle hit

        # Block collisions
        block_to_remove = None
        for block in self.blocks:
            if block["rect"].colliderect(ball_rect):
                block_to_remove = block
                
                # Calculate reward based on last paddle hit
                reward += 1.0  # Base reward for breaking a block
                if self.last_hit_type == 'risky':
                    reward += 2.0
                elif self.last_hit_type == 'safe':
                    reward -= 2.0
                
                self.last_hit_type = 'neutral' # Reset hit type after use

                # Simple bounce logic
                overlap_rect = ball_rect.clip(block["rect"])
                if overlap_rect.width > overlap_rect.height:
                    self.ball_vel.y *= -1
                else:
                    self.ball_vel.x *= -1
                
                self._create_particles(block["rect"].center, block["color"])
                # SFX: Block break
                break

        if block_to_remove:
            self.blocks.remove(block_to_remove)

        return reward

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_BASE_SPEED

    def _initialize_blocks(self):
        self.blocks = []
        for i in range(self.NUM_BLOCK_ROWS):
            for j in range(self.NUM_BLOCK_COLS):
                block_rect = pygame.Rect(
                    self.WALL_THICKNESS + j * self.BLOCK_WIDTH + self.BLOCK_SPACING / 2,
                    self.BLOCK_START_Y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING),
                    self.BLOCK_WIDTH - self.BLOCK_SPACING,
                    self.BLOCK_HEIGHT - self.BLOCK_SPACING,
                )
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": block_rect, "color": color})

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Damping
            p["radius"] *= 0.95
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0 and p["radius"] > 0.5]

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

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

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 20.0))))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color
            )

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball
        pos = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Draw score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.HEIGHT - 40))

        # Draw lives
        for i in range(self.lives):
            heart_pos_x = self.WIDTH - self.WALL_THICKNESS - 20 - (i * 30)
            heart_pos_y = self.HEIGHT - 28
            points = [
                (heart_pos_x, heart_pos_y + 5),
                (heart_pos_x - 10, heart_pos_y - 5),
                (heart_pos_x - 5, heart_pos_y - 10),
                (heart_pos_x, heart_pos_y - 5),
                (heart_pos_x + 5, heart_pos_y - 10),
                (heart_pos_x + 10, heart_pos_y - 5),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)

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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use Pygame for human interaction
    pygame.display.set_caption("Arcade Block Breaker")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the auto-advance rate
        
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()