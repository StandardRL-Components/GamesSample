
# Generated: 2025-08-28T00:52:34.579630
# Source Brief: brief_03930.md
# Brief Index: 3930

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    An intense, multi-ball pong-style arcade game. The player controls a paddle
    at the bottom of the screen and must survive for 60 seconds as three balls
    bounce around the play area. Missing a ball results in an instant game over.
    Players are rewarded for survival and for performing risky deflections by
    hitting the ball with the edges of the paddle.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle left and right. Survive for 60 seconds to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive 60 seconds of intense multi-ball pong action. Risky plays yield higher rewards."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Visuals
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_FG = (240, 240, 255)
        self.COLOR_PADDLE_GLOW = (100, 100, 255)
        self.COLOR_RISKY_PARTICLE = (255, 255, 100)
        self.COLOR_UI = (200, 200, 220)

        # Gameplay
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 10
        self.PADDLE_Y_POS = self.SCREEN_HEIGHT - 40

        self.BALL_RADIUS = 7
        self.BALL_SPEED = 5
        self.NUM_BALLS = 3

        self.RISKY_HIT_THRESHOLD_RATIO = 0.3 # 30% from edge

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)

        # --- State Variables ---
        self.paddle = None
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        # Initialize state
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize paddle
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y_POS,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize balls
        self.balls = []
        for _ in range(self.NUM_BALLS):
            self._spawn_ball()

        self.particles = []

        return self._get_observation(), self._get_info()

    def _spawn_ball(self):
        """Spawns a ball in the top half of the screen with a downward velocity."""
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)  # Downward cone
        speed = self.BALL_SPEED
        ball = {
            "pos": np.array([
                self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS),
                self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_HEIGHT / 2),
            ], dtype=np.float32),
            "vel": np.array([
                speed * math.cos(angle),
                speed * math.sin(angle)
            ], dtype=np.float32),
        }
        self.balls.append(ball)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward per frame

        # --- 1. Handle Input & Update Paddle ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # --- 2. Update Game State ---
        # Update balls and check collisions
        for ball in self.balls:
            ball["pos"] += ball["vel"]

            # Wall collisions
            if ball["pos"][0] <= self.BALL_RADIUS or ball["pos"][0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                ball["vel"][0] *= -1
                ball["pos"][0] = np.clip(ball["pos"][0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
                # SFX: wall_bounce.wav

            if ball["pos"][1] <= self.BALL_RADIUS:
                ball["vel"][1] *= -1
                ball["pos"][1] = self.BALL_RADIUS
                # SFX: wall_bounce.wav

            # Paddle collision
            ball_rect = pygame.Rect(ball["pos"][0] - self.BALL_RADIUS, ball["pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect) and ball["vel"][1] > 0:
                ball["vel"][1] *= -1.0
                ball["pos"][1] = self.paddle.top - self.BALL_RADIUS  # Prevent sticking

                dist_from_center = ball["pos"][0] - self.paddle.centerx
                normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
                
                # Apply spin based on hit location
                ball["vel"][0] += normalized_dist * 2.0
                
                # Re-normalize speed to prevent acceleration
                current_speed = np.linalg.norm(ball["vel"])
                if current_speed > 0:
                    ball["vel"] = (ball["vel"] / current_speed) * self.BALL_SPEED

                # Calculate reward based on risk
                risky_hit_threshold = self.PADDLE_WIDTH / 2 * self.RISKY_HIT_THRESHOLD_RATIO
                if abs(dist_from_center) > risky_hit_threshold:
                    hit_reward = 1.0
                    self._create_particles(ball["pos"], 25, self.COLOR_RISKY_PARTICLE, 2.5)
                    # SFX: risky_hit.wav
                else:
                    hit_reward = -0.2
                    self._create_particles(ball["pos"], 10, self.COLOR_FG, 1.5)
                    # SFX: safe_hit.wav
                
                reward += hit_reward
                self.score += hit_reward

            # Game over condition
            if ball["pos"][1] > self.SCREEN_HEIGHT:
                self.game_over = True
                # SFX: game_over.wav

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # --- 3. Check Termination ---
        self.steps += 1
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward += 100  # Win bonus
            self.score += 100
            # SFX: win.wav

        # --- 4. Return State ---
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, count, color, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_multiplier
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(15, 30),
                "color": color,
                "max_life": 30,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        self.clock.tick(self.FPS)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles first
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"]
            size = 2
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            temp_surf.fill(color + (alpha,))
            self.screen.blit(temp_surf, (int(p["pos"][0] - size/2), int(p["pos"][1] - size/2)))

        # Draw paddle with a subtle glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW + (50,), glow_surf.get_rect(), border_radius=6)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_FG, self.paddle, border_radius=3)

        # Draw balls with anti-aliasing
        for ball in self.balls:
            pos = (int(ball["pos"][0]), int(ball["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_FG)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_FG)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (20, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.2f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_surf, time_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play ---
    # This part requires a window and is for testing/demonstration.
    # It will not work in a purely headless environment.
    try:
        import os
        # Set a video driver that can open a window
        if os.name == "nt": # For Windows
             os.environ["SDL_VIDEODRIVER"] = "directx"
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Ball Pong")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # --- Action Mapping for Manual Control ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            action = [movement, 0, 0] # Space and Shift are not used

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Rendering ---
            # The observation is already a rendered frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

        print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")

    except Exception as e:
        print("\nCould not create Pygame window for manual play.")
        print("This is normal in a headless environment.")
        print(f"Error: {e}")
    finally:
        env.close()