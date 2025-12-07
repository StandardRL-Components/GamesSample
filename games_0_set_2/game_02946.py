
# Generated: 2025-08-27T21:54:12.151656
# Source Brief: brief_02946.md
# Brief Index: 2946

        
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
        "A fast-paced, grid-based block breaker. Destroy all blocks to win, but lose a life if the ball falls. Ball speed increases as you clear blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_STEPS = 10000
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 40)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            1: (0, 200, 0),   # Green
            3: (0, 100, 255), # Blue
            5: (220, 50, 50),  # Red
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_message = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.current_ball_speed = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.blocks_destroyed_count = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()
        
        self.blocks = self._create_blocks()
        self.particles = []
        
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        block_rows = 5
        block_cols = 12
        block_width = self.WIDTH // block_cols
        block_height = 20
        top_offset = 50
        
        for r in range(block_rows):
            for c in range(block_cols):
                points = 1 if r >= 3 else (3 if r >= 1 else 5)
                color = self.BLOCK_COLORS[points]
                block = {
                    "rect": pygame.Rect(
                        c * block_width,
                        top_offset + r * block_height,
                        block_width - 1,
                        block_height - 1
                    ),
                    "color": color,
                    "points": points
                }
                blocks.append(block)
        return blocks
    
    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # 1. Handle Input
        paddle_moved = self._handle_input(action)
        if paddle_moved:
            reward -= 0.02 # Small penalty for moving

        # 2. Update Game Logic
        self._update_particles()
        
        if self.ball_launched:
            ball_reward = self._update_ball()
            reward += ball_reward
        else:
            # Ball follows paddle if not launched
            self.ball_pos[0] = self.paddle.centerx

        # 3. Check for Termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = (random.random() - 0.5) * (math.pi / 3) # -30 to +30 degrees
            self.ball_vel = [
                self.current_ball_speed * math.sin(angle),
                -self.current_ball_speed * math.cos(angle)
            ]
            # sfx: launch_ball
        
        return moved

    def _update_ball(self):
        reward = 0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if ball_rect.left <= 0:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.right >= self.WIDTH:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            
            # Change angle based on hit location
            hit_pos = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            hit_pos = max(-0.95, min(0.95, hit_pos)) # Clamp to avoid extreme horizontal angles
            
            angle = math.asin(hit_pos)
            self.ball_vel[0] = self.current_ball_speed * math.sin(angle)
            self.ball_vel[1] = -self.current_ball_speed * math.cos(angle)
            # sfx: paddle_bounce

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                reward += 0.1 # Reward for any hit
                reward += block["points"]
                self.score += block["points"]
                
                # Create explosion particles
                self._create_explosion(block["rect"].center, block["color"])
                
                self.blocks.remove(block)
                # sfx: block_destroy
                
                # Simple reflection logic
                self.ball_vel[1] *= -1

                self.blocks_destroyed_count += 1
                if self.blocks_destroyed_count % 20 == 0:
                    self.current_ball_speed += 0.5
                    # Rescale velocity to new speed
                    speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                    if speed > 0:
                        self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
                        self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed

                break # Only handle one block collision per frame

        # Ball out of bounds
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            reward -= 1 # Penalty for losing a ball
            self._reset_ball()
            # sfx: lose_life
            
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        
        if self.lives <= 0:
            terminated = True
            terminal_reward = -100 # Lose penalty
        elif not self.blocks:
            terminated = True
            terminal_reward = 100 # Win bonus
            self.score += 100 # Add to visual score too
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return terminated, terminal_reward

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particle = {
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["vel"][1] += 0.1 # Gravity
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 20 - (i * 25), 22, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 20 - (i * 25), 22, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            message_text = self.font_message.render(message, True, self.COLOR_TEXT)
            text_rect = message_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(message_text, text_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()