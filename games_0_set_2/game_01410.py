
# Generated: 2025-08-27T17:02:59.056933
# Source Brief: brief_01410.md
# Brief Index: 1410

        
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
        "A retro arcade Breakout clone. Clear bricks to score points, but don't lose the ball! "
        "Hitting the ball with the edge of the paddle gives more control."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE_1 = (255, 255, 100)
    COLOR_PARTICLE_2 = (255, 180, 50)
    BRICK_COLORS = {
        10: (0, 200, 100),   # Green
        20: (0, 150, 255),   # Blue
        30: (255, 50, 100),  # Red
    }

    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    BALL_SPEED = 7
    MAX_LIVES = 5
    WIN_SCORE = 500
    MAX_STEPS = 2000
    BRICK_ROWS = 5
    BRICK_COLS = 12
    BRICK_WIDTH = 50
    BRICK_HEIGHT = 20
    BRICK_SPACING = 4
    BRICK_OFFSET_TOP = 50
    BRICK_OFFSET_LEFT = (SCREEN_WIDTH - (BRICK_COLS * (BRICK_WIDTH + BRICK_SPACING) - BRICK_SPACING)) // 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables (initialized in reset)
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.score = None
        self.lives = None
        self.steps = None
        self.game_over = None
        self.ball_launched = None
        self.consecutive_hits = None
        self.last_ball_y = None
        self.stuck_frames = None
        self.reward_this_step = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 20,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_launched = False

        self._generate_bricks()
        self.particles = []

        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.game_over = False
        self.consecutive_hits = 0
        
        self.last_ball_y = self.ball_pos.y
        self.stuck_frames = 0

        return self._get_observation(), self._get_info()

    def _generate_bricks(self):
        self.bricks = []
        brick_points = list(self.BRICK_COLORS.keys())
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                if self.np_random.random() > 0.1:  # 90% chance of a brick
                    x = self.BRICK_OFFSET_LEFT + col * (self.BRICK_WIDTH + self.BRICK_SPACING)
                    y = self.BRICK_OFFSET_TOP + row * (self.BRICK_HEIGHT + self.BRICK_SPACING)
                    rect = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                    points = self.np_random.choice(brick_points)
                    color = self.BRICK_COLORS[points]
                    self.bricks.append({"rect": rect, "points": points, "color": color})

    def step(self, action):
        self.reward_this_step = -0.01  # Time penalty

        if not self.game_over:
            self._handle_input(action)
            self._update_ball()
            self._handle_collisions()

        self.steps += 1
        terminated = self._check_termination()
        
        # Apply terminal rewards only once
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.reward_this_step += 100
            else: # Lost all lives or ran out of time
                self.reward_this_step -= 100
            self.game_over = True


        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        self.paddle_rect.x = np.clip(self.paddle_rect.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Ball Launch
        if not self.ball_launched and space_held:
            # sfx: launch_ball.wav
            self.ball_launched = True
            initial_angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = pygame.Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.BALL_SPEED

    def _update_ball(self):
        if self.ball_launched:
            self.ball_pos += self.ball_vel
            # Anti-softlock check
            if abs(self.ball_pos.y - self.last_ball_y) < 0.1:
                self.stuck_frames += 1
                if self.stuck_frames > 100:
                    self.ball_vel.y += self.np_random.choice([-0.5, 0.5])
                    self.stuck_frames = 0
            else:
                self.stuck_frames = 0
            self.last_ball_y = self.ball_pos.y
        else:
            self.ball_pos.x = self.paddle_rect.centerx
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: bounce_wall.wav
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
            # sfx: bounce_wall.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            # sfx: bounce_paddle.wav
            self.consecutive_hits = 0 # Missed hitting a brick
            
            # Calculate bounce angle based on impact point
            offset = (self.ball_pos.x - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            offset = np.clip(offset, -1, 1)
            
            # Reward for paddle hit type
            if abs(offset) < 0.1: # Central 20%
                self.reward_this_step -= 2
            
            bounce_angle = offset * (math.pi / 2.5) - (math.pi / 2)
            self.ball_vel = pygame.Vector2(math.cos(bounce_angle), math.sin(bounce_angle)) * self.BALL_SPEED
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS - 1 # Prevent sticking

        # Brick collisions
        hit_brick = None
        for brick in self.bricks:
            if ball_rect.colliderect(brick["rect"]):
                hit_brick = brick
                break
        
        if hit_brick:
            # sfx: break_brick.wav
            self.bricks.remove(hit_brick)
            self.score += hit_brick["points"]
            self.reward_this_step += 10
            self.consecutive_hits += 1
            if self.consecutive_hits > 1:
                self.reward_this_step += 5 # Consecutive hit bonus
            
            self._create_particles(hit_brick["rect"].center, hit_brick["color"])

            # Determine bounce direction
            # A simple approach: reverse vertical velocity
            self.ball_vel.y *= -1

        # Bottom wall (lose life)
        if self.ball_pos.y >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            # sfx: lose_life.wav
            self.lives -= 1
            self.reward_this_step -= 5
            self.consecutive_hits = 0
            if self.lives > 0:
                self.ball_launched = False
                self._reset_ball_position()
            else:
                self.game_over = True

    def _reset_ball_position(self):
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.last_ball_y = self.ball_pos.y
        self.stuck_frames = 0

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            lifespan = self.np_random.integers(10, 25)
            particle_color = random.choice([color, self.COLOR_PARTICLE_1, self.COLOR_PARTICLE_2])
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": particle_color})

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in brick["color"]), brick["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Ball glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_center = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, (100, 100, 200, 40))
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, int(255 * (p["lifespan"] / 25)))
                size = max(1, int(p["lifespan"] / 5))
                pygame.draw.rect(self.screen, p["color"] + (alpha,), (int(p["pos"].x), int(p["pos"].y), size, size))


    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Playable Demo ---
    # This part will not run in the headless testing environment
    # but is useful for local testing and visualization.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Breakout")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n" + "="*30)
        print(f"GAME: {env.game_description}")
        print(f"CONTROLS: {env.user_guide}")
        print("="*30 + "\n")
        
        while not done:
            movement = 0 # no-op
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1

            if keys[pygame.K_ESCAPE]:
                done = True

            action = [movement, space_held, 0] # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset() # Auto-restart
                pygame.time.wait(2000)

            # Draw the observation from the environment to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Match the intended FPS

    except Exception as e:
        print(f"An error occurred during the human-playable demo: {e}")
        print("This might be expected if running in a headless environment.")
    finally:
        env.close()