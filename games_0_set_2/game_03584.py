
# Generated: 2025-08-27T23:47:34.867623
# Source Brief: brief_03584.md
# Brief Index: 3584

        
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
    user_guide = "Controls: ←→ to move paddle. Press space to launch the ball."

    # Must be a short, user-facing description of the game:
    game_description = "A retro arcade block breaker. Clear all the blocks by bouncing the ball off your paddle."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.PADDLE_SPEED = 12
        self.BALL_MAX_SPEED = 7
        self.MAX_STEPS = 3000 # Increased from brief for better playability

        # Color Palette
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PADDLE = (0, 200, 255)
        self.COLOR_PADDLE_HL = (180, 240, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 255, 80),
            (80, 255, 80), (80, 160, 255), (160, 80, 255)
        ]

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
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_in_play = None
        self.blocks = None
        self.block_data = None
        self.balls_left = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.particles = []

        self._create_blocks()
        self._reset_paddle_and_ball()
        
        return self._get_observation(), self._get_info()
    
    def _reset_paddle_and_ball(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.ball_in_play = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)

    def _create_blocks(self):
        self.blocks = []
        self.block_data = []
        block_width = 50
        block_height = 20
        num_cols = 11
        num_rows = 6
        gap = 4
        
        total_block_width = num_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append(rect)
                color_index = i % len(self.BLOCK_COLORS)
                self.block_data.append({
                    "color": self.BLOCK_COLORS[color_index],
                    "value": (num_rows - i) # Higher blocks are worth more
                })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.01 # Small time penalty to encourage action
        
        # Update game logic
        self._update_paddle(movement)
        ball_hit_paddle, ball_hit_block, block_value = self._update_ball(space_held)
        self._update_particles()
        
        if ball_hit_paddle:
            reward += 0.1
        if ball_hit_block:
            reward += block_value
            self.score += block_value * 10
            # sfx: block break
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if not self.blocks: # Win condition
                reward += 50
                self.score += 1000
            elif self.balls_left < 0: # Lose condition
                reward -= 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

    def _update_ball(self, space_held):
        if not self.ball_in_play:
            if space_held:
                # sfx: launch ball
                self.ball_in_play = True
                initial_vx = (self.rng.random() - 0.5) * 2 # Random direction
                self.ball_vel = np.array([initial_vx, -self.BALL_MAX_SPEED * 0.8], dtype=float)
            else:
                self.ball_pos[0] = self.paddle.centerx
            return False, False, 0

        # Store previous position for collision response
        prev_pos = self.ball_pos.copy()
        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = prev_pos[0] # Prevent sticking
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: wall bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = prev_pos[1]
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # sfx: wall bounce

        # Bottom wall (lose ball)
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            # sfx: lose life
            if self.balls_left >= 0:
                self._reset_paddle_and_ball()
            else:
                self.game_over = True
            return False, False, 0

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle bounce
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.BALL_MAX_SPEED
            
            # Anti-stuck: ensure ball has some horizontal velocity
            if abs(self.ball_vel[0]) < 0.1:
                self.ball_vel[0] = 0.2 * np.sign(self.ball_vel[0] or 1)

            # Clamp speed
            speed = np.linalg.norm(self.ball_vel)
            if speed > self.BALL_MAX_SPEED:
                self.ball_vel = self.ball_vel / speed * self.BALL_MAX_SPEED
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Place ball on top of paddle
            return True, False, 0

        # Block collisions
        for i, block in reversed(list(enumerate(self.blocks))):
            if ball_rect.colliderect(block):
                block_data = self.block_data[i]
                self._create_explosion(block.center, block_data["color"])

                # Determine bounce direction
                # A simple but effective method: check overlap
                prev_ball_rect = pygame.Rect(prev_pos[0] - self.BALL_RADIUS, prev_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                if prev_ball_rect.bottom <= block.top or prev_ball_rect.top >= block.bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1

                self.blocks.pop(i)
                self.block_data.pop(i)
                return False, True, block_data["value"]

        return False, False, 0

    def _update_particles(self):
        # Ball trail
        if self.ball_in_play:
            p = {
                "pos": self.ball_pos.copy(),
                "vel": self.rng.random(2) * 0.5 - 0.25,
                "radius": self.BALL_RADIUS * 0.5,
                "color": self.COLOR_BALL_GLOW,
                "life": 10
            }
            self.particles.append(p)
        
        # Update all particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["radius"] *= 0.95
            p["life"] -= 1

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            p = {
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.rng.integers(2, 5),
                "color": color,
                "life": self.rng.integers(15, 30)
            }
            self.particles.append(p)

    def _check_termination(self):
        if self.game_over: # Already set by losing last ball
            return True
        if not self.blocks: # All blocks cleared
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_particles()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["life"] / 30))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*p["color"], alpha))

    def _render_blocks(self):
        for i, block in enumerate(self.blocks):
            color = self.block_data[i]["color"]
            darker_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, color, block, border_radius=3)
            pygame.draw.rect(self.screen, darker_color, block, width=2, border_radius=3)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        highlight_rect = self.paddle.copy()
        highlight_rect.height = 3
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_HL, highlight_rect, border_radius=4)

    def _render_ball(self):
        pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_BALL_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius-2, (*self.COLOR_BALL_GLOW, 70))
        
        # Ball
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        ball_text = self.font_main.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 150, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_PADDLE_HL)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_main.render(message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of the environment's functionality
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("Block Breaker - Manual Control")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(60) # Run at 60 FPS for smooth human play
        
    env.close()