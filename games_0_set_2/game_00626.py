
# Generated: 2025-08-27T14:15:25.111282
# Source Brief: brief_00626.md
# Brief Index: 626

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A procedurally generated block-breaking game where an RL agent learns to bounce a ball
    off a paddle to destroy blocks and maximize its score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Use the paddle to keep the ball in play, "
        "destroy all the blocks, and get the high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    INITIAL_BALLS = 3

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_WALL = (100, 100, 120)
    COLOR_TEXT = (200, 200, 220)
    BLOCK_COLORS = {
        1: (50, 200, 50),   # Green
        3: (50, 100, 220),  # Blue
        5: (220, 50, 50)    # Red
    }

    # Game Object Sizes
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED = 5
    WALL_THICKNESS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = False
        self.blocks = []
        self.particles = []

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        self._reset_ball()
        self._generate_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Frame Advance ---
        if self.auto_advance:
            self.clock.tick(30)
        
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Handle Input ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            # sound: ball_launch.wav
            initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [self.BALL_SPEED * math.cos(initial_angle), self.BALL_SPEED * math.sin(initial_angle)]

        # --- Update Game Logic ---
        if self.ball_launched:
            reward += self._update_ball()
        else:
            # Ball follows paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

        self._update_particles()
        
        # --- Check Termination ---
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
        
        if terminated:
            if not self.blocks: # Win condition
                reward += 50
            elif self.balls_left <= 0: # Lose condition
                reward -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        ball_x = self.paddle.centerx
        ball_y = self.paddle.top - self.BALL_RADIUS
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.center = (ball_x, ball_y)
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        num_cols = 10
        num_rows = 5
        col_width = (self.SCREEN_WIDTH - 2 * self.WALL_THICKNESS) / num_cols
        
        for row in range(num_rows):
            for col in range(num_cols):
                # Procedural generation: 80% chance of a block
                if self.np_random.random() < 0.8:
                    block_x = self.WALL_THICKNESS + col * col_width + (col_width - block_width) / 2
                    block_y = 60 + row * (block_height + 5)
                    
                    # Randomly assign score/color
                    rand_val = self.np_random.random()
                    if rand_val < 0.6: score = 1 # 60% chance
                    elif rand_val < 0.9: score = 3 # 30% chance
                    else: score = 5 # 10% chance

                    block_rect = pygame.Rect(block_x, block_y, block_width, block_height)
                    self.blocks.append({"rect": block_rect, "score": score, "color": self.BLOCK_COLORS[score]})

    def _update_ball(self):
        reward = 0
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= self.WALL_THICKNESS or self.ball.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            self.ball.left = max(self.ball.left, self.WALL_THICKNESS)
            self.ball.right = min(self.ball.right, self.SCREEN_WIDTH - self.WALL_THICKNESS)
            # sound: wall_bounce.wav
        if self.ball.top <= self.WALL_THICKNESS:
            self.ball_vel[1] *= -1
            self.ball.top = max(self.ball.top, self.WALL_THICKNESS)
            # sound: wall_bounce.wav

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            # Add "spin" based on where the ball hits the paddle
            hit_pos_norm = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_pos_norm * 2
            # Normalize speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED
            reward += 0.1
            # sound: paddle_hit.wav

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if self.ball.colliderect(block["rect"]):
                hit_block = block
                break
        
        if hit_block:
            self.blocks.remove(hit_block)
            reward += hit_block["score"]
            self.score += hit_block["score"]
            self._create_particles(hit_block["rect"].center, hit_block["color"])
            # sound: block_break.wav

            # Simple bounce logic: reverse vertical velocity
            self.ball_vel[1] *= -1

        # Check for win
        if not self.blocks:
            self.game_over = True

        # Check for ball loss
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 1
            # sound: lose_ball.wav
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color, "radius": radius})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["radius"] *= 0.95 # Shrink effect
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            # Add a subtle border for definition
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], width=2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball with antialiasing
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 20))
            color_with_alpha = (*p["color"], alpha)
            # Use a temporary surface for alpha blending
            particle_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(particle_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.SCREEN_HEIGHT - 35))

        # Balls left
        ball_text = self.font_large.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 35))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 50 + i * 20, self.SCREEN_HEIGHT - 22, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 50 + i * 20, self.SCREEN_HEIGHT - 22, 6, self.COLOR_BALL)
        
        # Game Over Message
        if self.game_over:
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(message, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        # We need to reset first to initialize surfaces
        self.reset()
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

# Example usage:
if __name__ == '__main__':
    # To run with human interaction
    import pygame
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if terminated:
            # Game over, wait for reset
            pygame.time.wait(100)
            continue
            
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game

        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Frame Rate ---
        # The environment's internal clock handles this, so we don't need another clock.tick() here
        
    env.close()
    pygame.quit()