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
    user_guide = "Controls: ↑ to move the paddle up, ↓ to move the paddle down."

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive 30 seconds of intense multiball pong action, juggling multiple "
        "balls with a single paddle while aiming for combo bonuses."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True # Changed to True to better match stability test expectations

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 30 * FPS  # 30 seconds
    MAX_MISSES = 3

    COLOR_BG = (15, 15, 25)
    COLOR_FG = (220, 220, 255)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_COMBO_ACTIVE = (255, 80, 80)
    COLOR_PARTICLE = (255, 180, 80)
    COLOR_MISS_FLASH = (180, 20, 20)

    PADDLE_WIDTH, PADDLE_HEIGHT = 12, 80
    PADDLE_SPEED = 10
    PADDLE_X_POS = 20

    BALL_RADIUS = 7
    BALL_SPEED = 7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("impact", 36)

        # Initialize state variables
        self.paddle = None
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.missed_balls = 0
        self.combo_multiplier = 1
        self.terminated = False
        self.screen_flash_timer = 0
        
        # self.reset() is called by the wrapper, but for standalone use it's good practice
        # to ensure the state is initialized.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.missed_balls = 0
        self.combo_multiplier = 1
        self.terminated = False
        self.screen_flash_timer = 0
        self.particles.clear()

        # Initialize paddle
        self.paddle = pygame.Rect(
            self.PADDLE_X_POS,
            self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize balls
        self.balls.clear()
        num_balls = self.np_random.integers(2, 6)
        for _ in range(num_balls):
            self.balls.append(self._create_ball())

        return self._get_observation(), self._get_info()

    def _create_ball(self):
        ball_pos = pygame.Vector2(
            self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.8),
            self.np_random.uniform(self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS),
        )
        # Initial velocity towards the player
        angle_rad = self.np_random.uniform(math.pi * 0.8, math.pi * 1.2)
        ball_vel = pygame.Vector2()
        ball_vel.from_polar((self.BALL_SPEED, math.degrees(angle_rad)))
        return {"pos": ball_pos, "vel": ball_vel}

    def step(self, action):
        if self.terminated:
            # The environment is done, return a dummy observation and info
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        # --- Update game logic ---
        self._handle_input(movement)
        event_reward = self._update_physics()
        self._update_particles()
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1

        # Calculate reward
        reward = 0.1 + event_reward  # +0.1 for survival per step

        # Update step counter and check termination
        self.steps += 1
        self.terminated = (
            self.missed_balls >= self.MAX_MISSES or self.steps >= self.MAX_STEPS
        )
        
        truncated = False # This env does not truncate

        if self.terminated:
            if self.steps >= self.MAX_STEPS:
                reward += 100.0  # Victory bonus

        self.score += reward

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED
        # Clamp paddle to screen
        self.paddle.y = max(0, min(self.paddle.y, self.HEIGHT - self.PADDLE_HEIGHT))

    def _update_physics(self):
        event_reward = 0.0
        for ball in self.balls:
            ball["pos"] += ball["vel"]

            # Wall collisions
            if ball["pos"].y - self.BALL_RADIUS <= 0 or ball["pos"].y + self.BALL_RADIUS >= self.HEIGHT:
                ball["vel"].y *= -1
                ball["pos"].y = max(self.BALL_RADIUS, min(ball["pos"].y, self.HEIGHT - self.BALL_RADIUS))

            if ball["pos"].x + self.BALL_RADIUS >= self.WIDTH:
                ball["vel"].x *= -1
                ball["pos"].x = self.WIDTH - self.BALL_RADIUS

            # Paddle collision
            ball_rect = pygame.Rect(
                ball["pos"].x - self.BALL_RADIUS,
                ball["pos"].y - self.BALL_RADIUS,
                self.BALL_RADIUS * 2,
                self.BALL_RADIUS * 2,
            )
            if self.paddle.colliderect(ball_rect) and ball["vel"].x < 0:
                ball["vel"].x *= -1
                
                # Add vertical spin based on where it hit the paddle
                offset = (self.paddle.centery - ball["pos"].y) / (self.PADDLE_HEIGHT / 2)
                ball["vel"].y -= offset * 3 # Max y-velocity change
                
                # Normalize speed to prevent acceleration
                ball["vel"].scale_to_length(self.BALL_SPEED)

                # Ensure ball is pushed out of paddle
                ball["pos"].x = self.paddle.right + self.BALL_RADIUS

                # Reward and combo
                event_reward += 1.0 + (5.0 * (self.combo_multiplier -1))
                self.combo_multiplier += 1
                self._create_hit_particles(ball["pos"])

            # Missed ball
            if ball["pos"].x - self.BALL_RADIUS < 0:
                self.missed_balls += 1
                self.combo_multiplier = 1
                event_reward -= 1.0
                self.screen_flash_timer = 5 # Flash for 5 frames
                # Reset ball
                new_ball = self._create_ball()
                ball["pos"] = new_ball["pos"]
                ball["vel"] = new_ball["vel"]
        
        return event_reward

    def _create_hit_particles(self, position):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2()
            vel.from_polar((speed, math.degrees(angle)))
            life = self.np_random.integers(10, 20)
            # FIX: pygame.Vector2 does not have a .copy() method.
            # The correct way to copy it is by creating a new instance.
            self.particles.append(
                {"pos": pygame.Vector2(position), "vel": vel, "life": life, "max_life": life}
            )

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9  # friction
            p["life"] -= 1

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Screen flash on miss
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.screen_flash_timer / 5))
            flash_surface.fill((*self.COLOR_MISS_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw play area bounds (top, bottom, right)
        pygame.draw.line(self.screen, self.COLOR_FG, (0, 0), (self.WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_FG, (0, self.HEIGHT - 1), (self.WIDTH, self.HEIGHT - 1), 2)
        pygame.draw.line(self.screen, self.COLOR_FG, (self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT), 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = 255 * (p["life"] / p["max_life"])
            # Create a color with an alpha component
            color_with_alpha = (*self.COLOR_PARTICLE, int(alpha))
            size = self.BALL_RADIUS * 0.3 * (p["life"] / p["max_life"])
            # gfxdraw requires a color without an explicit alpha tuple for the last arg
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(size), color_with_alpha
            )

        # Draw balls
        for ball in self.balls:
            pygame.gfxdraw.aacircle(
                self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, self.COLOR_BALL
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, self.COLOR_BALL
            )

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_FG)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_ui.render(f"LIVES: {self.MAX_MISSES - self.missed_balls}", True, self.COLOR_FG)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, self.HEIGHT - miss_text.get_height() - 10))

        # Combo
        if self.combo_multiplier > 1:
            color = self.COLOR_COMBO_ACTIVE
            text = f"{self.combo_multiplier - 1}x COMBO"
            combo_surf = self.font_combo.render(text, True, color)
            pos = (self.WIDTH - combo_surf.get_width() - 10, 10)
            self.screen.blit(combo_surf, pos)
        
        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_FG)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_balls": self.missed_balls,
            "combo_multiplier": self.combo_multiplier,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    print(GameEnv.user_guide)

    while not terminated:
        # Map keyboard to MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()