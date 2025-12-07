import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
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
        "A retro block-breaking game. Bounce the ball to destroy all blocks for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500  # Increased to allow for more complex plays

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_WALL = (100, 100, 120)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 0, 64)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = {
        1: (0, 200, 100),  # Green
        3: (50, 100, 255),  # Blue
        5: (255, 50, 50),   # Red
    }

    # Game Object Properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED = 5
    WALL_THICKNESS = 10
    
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = (SCREEN_WIDTH - 2 * WALL_THICKNESS) // BLOCK_COLS
    BLOCK_HEIGHT = 20
    BLOCK_AREA_TOP = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.game_over = False
        self.paddle_still_frames = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.paddle_still_frames = 0

        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        self.ball_launched = False
        self._reset_ball()

        self.blocks = []
        point_values = [5, 5, 3, 3, 1]
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                points = point_values[i]
                block_rect = pygame.Rect(
                    self.WALL_THICKNESS + j * self.BLOCK_WIDTH,
                    self.BLOCK_AREA_TOP + i * self.BLOCK_HEIGHT,
                    self.BLOCK_WIDTH - 1,
                    self.BLOCK_HEIGHT - 1
                )
                self.blocks.append({"rect": block_rect, "points": points, "color": self.BLOCK_COLORS[points]})
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    
    def step(self, action):
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            self.paddle_still_frames = 0
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            self.paddle_still_frames = 0
        else:
            self.paddle_still_frames += 1

        self.paddle.left = max(self.WALL_THICKNESS, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH - self.WALL_THICKNESS, self.paddle.right)

        if space_pressed and not self.ball_launched:
            # Play launch sound
            self.ball_launched = True
            # FIX: np.random.uniform requires low < high. Swapped arguments.
            angle = self.np_random.uniform(-2*math.pi/3, -math.pi/3) # Launch upwards
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.BALL_SPEED
            assert np.linalg.norm(self.ball_vel) > 0

        # --- Game Logic ---
        if self.ball_launched:
            reward += 0.01 # Small reward for keeping ball in play
            self.ball_pos += self.ball_vel
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collision
            if ball_rect.left <= self.WALL_THICKNESS:
                self.ball_vel[0] *= -1
                ball_rect.left = self.WALL_THICKNESS
            if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
                self.ball_vel[0] *= -1
                ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
            if ball_rect.top <= 0:
                self.ball_vel[1] *= -1
                ball_rect.top = 0
            
            # Paddle collision
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += offset * 2.0
                
                speed = np.linalg.norm(self.ball_vel)
                if speed > 0:
                    self.ball_vel = (self.ball_vel / speed) * self.BALL_SPEED
                
                ball_rect.bottom = self.paddle.top

            # Block collision
            hit_block = None
            for block in self.blocks:
                if ball_rect.colliderect(block["rect"]):
                    hit_block = block
                    break
            
            if hit_block:
                self.blocks.remove(hit_block)
                reward += hit_block["points"]
                self.score += hit_block["points"]
                self.ball_vel[1] *= -1
                self._create_particles(hit_block["rect"].center, hit_block["color"])
            
            self.ball_pos[0] = ball_rect.centerx
            self.ball_pos[1] = ball_rect.centery

            # Lose ball
            if ball_rect.top >= self.SCREEN_HEIGHT:
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    reward -= 100
        else: # Ball not launched, stick to paddle
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # Paddle stillness penalty
        if self.paddle_still_frames > 5:
            reward -= 0.02

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        self.steps += 1
        terminated = self._check_termination()
        if terminated and len(self.blocks) == 0:
            reward += 100
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.blocks:
            self.game_over = True
            return True
        return False

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = self.np_random.uniform(-2, 2, size=2).astype(np.float64)
            self.particles.append({"pos": np.array(pos, dtype=np.float64), "vel": vel, "life": 20, "color": color})
            
    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Background gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 20))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill((*p["color"], alpha))
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        self._render_ui()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, 10))

        # Balls left
        balls_text = self.font_main.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 10, 10))

        if not self.ball_launched and self.balls_left > 0:
            prompt_text = self.font_main.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            text_rect = prompt_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 100))
            self.screen.blit(prompt_text, text_rect)

        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    import time
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # Set up a display window
    pygame.display.set_caption("Block Breaker")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    total_reward = 0
    
    while not terminated and not truncated:
        # Map keyboard inputs to actions
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # unused
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        env.clock.tick(30) # Control the frame rate

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    time.sleep(3) # Pause to see the final screen
    env.close()