
# Generated: 2025-08-28T04:36:13.125873
# Source Brief: brief_05304.md
# Brief Index: 5304

        
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
        "Controls: ← and → to move the paddle. Keep the ball in play and break all the bricks!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade classic. Control a paddle to bounce a ball, break bricks, and chase a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto-advance mode
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 500
        self.STARTING_LIVES = 3

        # Colors (Bright and Contrasting)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.BRICK_COLORS = {
            1: (0, 200, 100), # Green
            2: (0, 150, 255), # Blue
            3: (255, 80, 80), # Red
        }

        # Game Object Properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5.0
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup (Headless) ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []
        self.last_speed_increase_score = 0
        
        # Initialize state
        self.reset()

        # Validate implementation after setup
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        self.last_speed_increase_score = 0

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        
        # Start ball with a random upward angle
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.INITIAL_BALL_SPEED

        self._generate_bricks()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_bricks(self):
        self.bricks = []
        brick_width, brick_height = 58, 20
        gap = 4
        rows, cols = 5, 10
        
        for r in range(rows):
            for c in range(cols):
                points = (rows - r) // 2 + 1  # 3, 3, 2, 2, 1
                color = self.BRICK_COLORS[points]
                brick = pygame.Rect(
                    c * (brick_width + gap) + gap + 10,
                    r * (brick_height + gap) + 40,
                    brick_width,
                    brick_height
                )
                self.bricks.append({"rect": brick, "points": points, "color": color})

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        # --- Game Logic ---
        self._update_ball()
        brick_reward = self._handle_collisions()
        self._update_particles()
        
        # --- Reward Calculation ---
        reward = 0.01 + brick_reward # Small reward for surviving the frame
        
        # --- Termination Check ---
        terminated = False
        if self.lives <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.score >= self.WIN_SCORE:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        brick_reward = 0

        # Walls
        if ball_rect.left <= 0:
            self.ball_vel.x *= -1
            ball_rect.left = 0
            # sfx: wall_bounce
        if ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            ball_rect.right = self.WIDTH
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            ball_rect.top = 0
            # sfx: wall_bounce

        # Out of bounds (bottom)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life
            if self.lives > 0:
                self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.INITIAL_BALL_SPEED * (1 + (self.score // 100) * 0.1)

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # sfx: paddle_bounce
            self.ball_vel.y *= -1
            
            # Add horizontal influence based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2
            
            # Normalize to maintain speed
            self.ball_vel.normalize_ip()
            current_speed = self.INITIAL_BALL_SPEED * (1 + (self.score // 100) * 0.1)
            self.ball_vel *= current_speed
            
            ball_rect.bottom = self.paddle.top

        # Bricks
        hit_brick = None
        for brick_data in self.bricks:
            if ball_rect.colliderect(brick_data["rect"]):
                hit_brick = brick_data
                break
        
        if hit_brick:
            # sfx: brick_break
            self.bricks.remove(hit_brick)
            self.score += hit_brick["points"]
            brick_reward += hit_brick["points"]
            
            self._create_particles(hit_brick["rect"].center, hit_brick["color"])

            # Determine bounce direction (simple approach)
            # Check overlap to decide if it's a vertical or horizontal hit
            dy_top = abs(ball_rect.bottom - hit_brick["rect"].top)
            dy_bottom = abs(ball_rect.top - hit_brick["rect"].bottom)
            dx_left = abs(ball_rect.right - hit_brick["rect"].left)
            dx_right = abs(ball_rect.left - hit_brick["rect"].right)
            
            min_overlap = min(dy_top, dy_bottom, dx_left, dx_right)

            if min_overlap == dy_top or min_overlap == dy_bottom:
                self.ball_vel.y *= -1
            else:
                self.ball_vel.x *= -1

            # Increase ball speed every 100 points
            if self.score // 100 > self.last_speed_increase_score // 100:
                self.last_speed_increase_score = self.score
                self.ball_vel.normalize_ip()
                new_speed = self.INITIAL_BALL_SPEED * (1 + (self.score // 100) * 0.1)
                self.ball_vel *= new_speed
                # sfx: speed_up
        
        self.ball_pos.x = ball_rect.centerx
        self.ball_pos.y = ball_rect.centery

        return brick_reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, 5))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, 5, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - 5, 0, 5, self.HEIGHT))

        # Draw Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"], border_radius=3)
            
        # Draw Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw Ball with a glow
        x, y = int(self.ball_pos.x), int(self.ball_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw Particles
        for p in self.particles:
            size = max(1, int(p["lifespan"] / 4))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y), size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Lives
        life_icon_surf = pygame.Surface((self.PADDLE_WIDTH / 4, self.PADDLE_HEIGHT / 2))
        life_icon_surf.fill(self.COLOR_PADDLE)
        for i in range(self.lives):
            self.screen.blit(life_icon_surf, (self.WIDTH - 20 - (i + 1) * (life_icon_surf.get_width() + 5), 15))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 0) if self.score >= self.WIN_SCORE else (255, 0, 0)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value to run without a display
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To run headlessly and save a video ---
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, 'video', episode_trigger=lambda x: x % 1 == 0) # record every episode
    
    # --- To play interactively ---
    try:
        interactive_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Arcade Breakout")
        
        obs, info = env.reset()
        terminated = False
        
        print(env.user_guide)
        
        while not terminated:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            action = [movement, 0, 0] # space and shift are not used
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation to the interactive screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            interactive_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
        print(f"Game Over. Final Score: {info['score']}")
        
        # Keep window open for a bit to see the final screen
        pygame.time.wait(2000)

    finally:
        env.close()