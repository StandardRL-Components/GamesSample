# Generated: 2025-08-27T14:33:47.204901
# Source Brief: brief_00724.md
# Brief Index: 724

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle. Try to destroy all the blocks with the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game where you control a paddle to bounce a ball and destroy a grid of blocks. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 4.0
        self.BLOCK_ROWS, self.BLOCK_COLS = 5, 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 18
        self.BLOCK_SPACING = 6
        self.UI_HEIGHT = 40

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150, 100)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_HEART = (255, 50, 50)
        self.BLOCK_COLORS = [
            (255, 70, 70), (255, 165, 0), (255, 255, 0),
            (0, 200, 0), (0, 128, 255), (148, 0, 211)
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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.win = None
        self.blocks_destroyed_count = None
        
        # self.reset() is called to set the initial state
        # self.validate_implementation() is a helper for development
        # In a typical gym env, these would not be in __init__
        obs, info = self.reset()
        # self.validate_implementation() # This can be commented out for production
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state variables first
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.blocks_destroyed_count = 0
        
        # Initialize game objects
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()
        
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) // 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.UI_HEIGHT + 20 + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT), "color": color})

        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
        # Recalculate speed on reset, depends on blocks_destroyed_count
        speed = self.INITIAL_BALL_SPEED + (self.blocks_destroyed_count // 10) * 0.4
        self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]

    def step(self, action):
        reward = -0.001 # Small penalty for existing
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right
        
        # Update paddle position
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        # Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Ball collision with walls
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.ball_pos[0], self.WIDTH - self.BALL_RADIUS))
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS + self.UI_HEIGHT:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS + self.UI_HEIGHT
            # sfx: wall_bounce

        # Ball collision with paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            
            # Change angle based on hit position
            hit_offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            new_angle = math.acos(self.ball_vel[0] / self._get_ball_speed()) - hit_offset * 0.8
            new_angle = max(-math.pi * 0.9, min(-math.pi * 0.1, new_angle)) # Clamp angle
            
            speed = self._get_ball_speed()
            self.ball_vel[0] = math.cos(new_angle) * speed
            self.ball_vel[1] = -abs(math.sin(new_angle) * speed)
            # sfx: paddle_hit

        # Ball collision with blocks
        block_hit = None
        for i, block_data in enumerate(self.blocks):
            if ball_rect.colliderect(block_data["rect"]):
                block_hit = i
                break
        
        if block_hit is not None:
            hit_block = self.blocks.pop(block_hit)
            self.score += 10
            reward += 1.0
            self.blocks_destroyed_count += 1
            # sfx: block_destroy
            
            # Create particles
            for _ in range(15):
                self.particles.append(self._create_particle(hit_block["rect"].center, hit_block["color"]))

            # Reverse ball velocity
            self.ball_vel[1] *= -1
            
            # Increase ball speed every 10 blocks
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                self._increase_ball_speed()

        # Ball out of bounds
        if self.ball_pos[1] > self.HEIGHT:
            self.lives -= 1
            reward -= 10.0
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
                self.win = False

        # Update particles
        self._update_particles()
        
        # Check termination conditions
        self.steps += 1
        terminated = False
        truncated = False
        if self.lives <= 0:
            self.game_over = True
            terminated = True
        elif not self.blocks:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_ball_speed(self):
        return math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)

    def _increase_ball_speed(self):
        speed = self._get_ball_speed()
        if speed == 0: return
        new_speed = speed + 0.4
        self.ball_vel[0] = (self.ball_vel[0] / speed) * new_speed
        self.ball_vel[1] = (self.ball_vel[1] / speed) * new_speed
        # sfx: speed_up

    def _create_particle(self, pos, color):
        return {
            "pos": list(pos),
            "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
            "lifetime": self.np_random.integers(15, 30),
            "color": color,
            "radius": self.np_random.uniform(1, 4)
        }
        
    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]),
                int(p["radius"]), (*p["color"], alpha)
            )

        # Render blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block_data["color"]), block_data["rect"], 1)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.BALL_RADIUS*2, self.BALL_RADIUS*2, self.BALL_RADIUS*2, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (ball_x - self.BALL_RADIUS*2, ball_y - self.BALL_RADIUS*2))
        
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # UI background
        pygame.draw.rect(self.screen, (0,0,0), (0, 0, self.WIDTH, self.UI_HEIGHT))
        
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))
        
        # Lives
        for i in range(self.lives):
            heart_pos = (self.WIDTH - 30 - i * 35, 8)
            pygame.gfxdraw.filled_polygon(self.screen, [
                (heart_pos[0] + 12, heart_pos[1] + 5), (heart_pos[0] + 24, heart_pos[1] + 15),
                (heart_pos[0] + 12, heart_pos[1] + 25), (heart_pos[0], heart_pos[1] + 15)
            ], self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 6, heart_pos[1] + 10, 8, self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 18, heart_pos[1] + 10, 8, self.COLOR_HEART)

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Add a semi-transparent background for readability
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)

            self.screen.blit(text_surf, text_rect)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        # Test specific game mechanics
        self.reset()
        initial_speed = self._get_ball_speed()
        assert math.isclose(initial_speed, self.INITIAL_BALL_SPEED), "Ball speed incorrect at start"
        
        self.blocks_destroyed_count = 9
        self._increase_ball_speed() # Should not trigger
        assert math.isclose(self._get_ball_speed(), initial_speed), "Ball speed increased too early"

        self.blocks_destroyed_count = 10
        self._increase_ball_speed() # Should trigger
        assert self._get_ball_speed() > initial_speed, "Ball speed did not increase at 10 blocks"

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # To run in a window, comment out the os.environ line at the top
    # and change the render_mode to "human"
    env = GameEnv(render_mode="rgb_array")
    
    # To play the game manually
    import pygame
    
    # If you want to see the game, you need a display.
    # For headless execution, this block can be removed.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Breakout")
        is_headless = False
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        is_headless = True

    obs, info = env.reset()
    done = False
    
    action = np.array([0, 0, 0]) # No-op, no space, no shift
    
    running = True
    while running:
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            
            # Reset action
            action[0] = 0 # No movement
            
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_r]: # Press 'r' to reset
                obs, info = env.reset()
                done = False

        if done: # If game is over, reset it
            obs, info = env.reset()
            done = False

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not is_headless:
            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(60) # Control the frame rate
        
        # Example of headless run: break after some steps
        if is_headless and env.steps > 1000:
            running = False
            print("Headless run finished.")

    env.close()