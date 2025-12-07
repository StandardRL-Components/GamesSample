import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based Breakout variant where the player controls a rotating paddle
    at the bottom of the screen. The goal is to destroy all blocks by hitting
    them with a ball. The game emphasizes strategic aiming and rewards
    chain reactions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑/↓ to rotate paddle. Press Space to launch the ball."
    )

    # User-facing description of the game
    game_description = (
        "A geometric Breakout variant. Aim your paddle and launch the ball to clear all the blocks."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.INITIAL_BALLS = 3

        # Paddle
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 10
        self.PADDLE_Y = self.HEIGHT - 35
        self.PADDLE_ROTATION_SPEED = 4  # degrees per frame
        self.MAX_PADDLE_ANGLE = 75  # degrees

        # Ball
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8

        # Blocks
        self.BLOCK_COLS, self.BLOCK_ROWS = 12, 5
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 48, 15
        self.BLOCK_SPACING = 4
        self.BLOCK_START_Y = 50
        self.TOTAL_BLOCKS = self.BLOCK_COLS * self.BLOCK_ROWS

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_UI = (240, 240, 240)
        self.BLOCK_COLORS = {
            1: (0, 200, 100),  # Green
            2: (0, 150, 255),  # Blue
            3: (255, 80, 80),   # Red
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ball_icon = pygame.font.SysFont("Arial", 16, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.paddle_pos = None
        self.paddle_angle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_active = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.blocks_hit_this_launch = None
        
        # Initialize RNG
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_pos = pygame.Vector2(self.WIDTH / 2, self.PADDLE_Y)
        self.paddle_angle = 0.0

        self.ball_active = False
        self._reset_ball_position()

        self.blocks = []
        for row in range(self.BLOCK_ROWS):
            for col in range(self.BLOCK_COLS):
                block_x = col * (self.BLOCK_WIDTH + self.BLOCK_SPACING) + (self.WIDTH - self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) + self.BLOCK_SPACING) / 2
                block_y = self.BLOCK_START_Y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                
                # Assign points/color based on row
                points = 1 if row >= 3 else (2 if row >= 1 else 3)
                color = self.BLOCK_COLORS[points]
                
                self.blocks.append({
                    "rect": pygame.Rect(block_x, block_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    "points": points,
                    "color": color
                })

        self.particles = []
        self.balls_left = self.INITIAL_BALLS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_hit_this_launch = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        collision_reward = self._update_ball()
        reward += collision_reward
        
        # Chain reaction bonus
        if self.blocks_hit_this_launch == 2:
            reward += 5.0 # Awarded once when the second block is hit

        self._update_particles()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.blocks: # Win condition
            reward += 100.0
        elif terminated and self.balls_left <= 0: # Loss condition
            reward -= 100.0

        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1

        # Paddle rotation
        if movement == 1:  # Rotate left
            self.paddle_angle -= self.PADDLE_ROTATION_SPEED
        elif movement == 2:  # Rotate right
            self.paddle_angle += self.PADDLE_ROTATION_SPEED
        self.paddle_angle = np.clip(self.paddle_angle, -self.MAX_PADDLE_ANGLE, self.MAX_PADDLE_ANGLE)

        # Ball launch
        if space_held and not self.ball_active:
            self.ball_active = True
            self.blocks_hit_this_launch = 0
            angle_rad = math.radians(self.paddle_angle - 90) # -90 because 0 angle is horizontal
            self.ball_vel = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.BALL_SPEED

    def _update_ball(self):
        if not self.ball_active:
            self._reset_ball_position()
            return 0.0

        reward = 0.0
        
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT)

        # Lose ball
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.balls_left -= 1
            self.ball_active = False
            self._reset_ball_position()
            if self.balls_left <= 0:
                self.game_over = True
            return -10.0 # Penalty for losing a ball

        # Paddle collision
        paddle_rect_surf = pygame.Surface((self.PADDLE_WIDTH, self.PADDLE_HEIGHT), pygame.SRCALPHA)
        paddle_rect = paddle_rect_surf.get_rect(center=self.paddle_pos)
        rotated_paddle_surf = pygame.transform.rotate(paddle_rect_surf, self.paddle_angle)
        rotated_paddle_rect = rotated_paddle_surf.get_rect(center=self.paddle_pos)

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        if rotated_paddle_rect.colliderect(ball_rect) and self.ball_vel.y > 0:
            angle_rad = math.radians(self.paddle_angle - 90)
            normal = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad))
            self.ball_vel = self.ball_vel.reflect(normal.rotate(90))
            self.ball_pos.y = min(self.ball_pos.y, self.PADDLE_Y - self.PADDLE_HEIGHT/2 - self.BALL_RADIUS - 1)
            self.blocks_hit_this_launch = 0 # Reset chain on paddle hit

        # Block collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                self.blocks.remove(block)
                self.score += block["points"]
                reward += 0.1 + block["points"] # Touch reward + destroy reward
                self.blocks_hit_this_launch += 1
                
                # Determine collision side for reflection
                dx = self.ball_pos.x - block["rect"].centerx
                dy = self.ball_pos.y - block["rect"].centery
                w, h = block["rect"].width / 2, block["rect"].height / 2
                
                if abs(dx / w) > abs(dy / h): # Horizontal collision
                    self.ball_vel.x *= -1
                else: # Vertical collision
                    self.ball_vel.y *= -1

                self._create_particles(block["rect"].center, block["color"])
                break # Handle one block at a time

        return reward

    def _reset_ball_position(self):
        angle_rad = math.radians(self.paddle_angle - 90)
        offset_y = self.BALL_RADIUS + self.PADDLE_HEIGHT / 2 + 2
        self.ball_pos = self.paddle_pos + pygame.Vector2(0, -offset_y).rotate(-self.paddle_angle)
        self.ball_vel = pygame.Vector2(0, 0)
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": random.uniform(2, 5),
                "lifespan": random.randint(15, 30),
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if not self.blocks or self.balls_left <= 0:
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Draw paddle
        paddle_surf = pygame.Surface((self.PADDLE_WIDTH, self.PADDLE_HEIGHT), pygame.SRCALPHA)
        paddle_surf.fill(self.COLOR_PADDLE)
        rotated_paddle_surf = pygame.transform.rotate(paddle_surf, self.paddle_angle)
        rotated_paddle_rect = rotated_paddle_surf.get_rect(center=self.paddle_pos)
        self.screen.blit(rotated_paddle_surf, rotated_paddle_rect)

        # Draw ball
        if self.ball_pos:
            ball_x, ball_y = int(self.ball_pos.x), int(self.ball_pos.y)
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color_with_alpha = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (p["pos"].x - p["radius"], p["pos"].y - p["radius"]))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            ball_icon_pos = (self.WIDTH - 20 - (i * 25), 20)
            pygame.gfxdraw.filled_circle(self.screen, ball_icon_pos[0], ball_icon_pos[1], 8, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, ball_icon_pos[0], ball_icon_pos[1], 8, self.COLOR_PADDLE)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # Example usage: run the game with random actions
    # This part requires a display. Set SDL_VIDEODRIVER to a valid driver.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human viewing
    pygame.display.set_caption("GridBreak")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        # Get action from keyboard for interactive play
        keys = pygame.key.get_pressed()
        mov = 0 # No-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)

    env.close()