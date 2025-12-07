import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade game.
    The player controls a paddle to bounce a ball and break oscillating blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Game Metadata ---
    game_description = "A retro arcade game where you control a paddle to bounce a ball and break oscillating blocks."
    user_guide = "Controls: Use ← and → arrow keys to move the paddle."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 50
    MAX_STEPS = 60 * FPS  # 60 seconds

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_PADDLE_GLOW = (100, 100, 200)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (150, 150, 255)
    COLOR_UI_TEXT = (200, 200, 220)
    BLOCK_COLORS = [
        (60, 160, 255),  # Blue
        (80, 220, 80),   # Green
        (255, 200, 50),  # Yellow
        (255, 80, 80),   # Red
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)

        # --- Game Object Properties ---
        self.paddle_width = 100
        self.paddle_height = 12
        self.paddle_speed = 8
        self.ball_radius = 6
        self.initial_ball_speed = 4

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.particles = []
        self.time_remaining = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        # Player Paddle
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.paddle_width) / 2,
            self.SCREEN_HEIGHT - self.paddle_height - 10,
            self.paddle_width,
            self.paddle_height
        )

        # Ball
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.ball_radius - 1], dtype=np.float32)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)  # Upwards angle
        self.ball_speed = self.initial_ball_speed
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.ball_speed

        # Blocks
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 4
        cols = 11
        for i in range(rows):
            for j in range(cols):
                block_rect = pygame.Rect(
                    j * (block_width + 10) + 35,
                    i * (block_height + 10) + 40,
                    block_width,
                    block_height
                )
                self.blocks.append({
                    "rect": block_rect,
                    "color": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)],
                    "alive": True
                })

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # --- Handle Actions ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += self.paddle_speed

        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.paddle_width, self.paddle.x))

        # --- Update Ball ---
        self.ball_pos += self.ball_vel

        # --- Collisions ---
        # Walls
        if self.ball_pos[0] - self.ball_radius <= 0 or self.ball_pos[0] + self.ball_radius >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.ball_radius, self.SCREEN_WIDTH - self.ball_radius)

        if self.ball_pos[1] - self.ball_radius <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.ball_radius, self.SCREEN_HEIGHT - self.ball_radius)

        # Paddle
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_radius, self.ball_pos[1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            reward += 0.1

            # Bounce logic for better control
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.paddle_width / 2)
            bounce_angle = offset * (math.pi / 2.6)  # Max angle ~70 degrees
            
            new_vel_x = self.ball_speed * math.sin(bounce_angle)
            new_vel_y = -self.ball_speed * math.cos(bounce_angle)
            
            self.ball_vel = np.array([new_vel_x, new_vel_y], dtype=np.float32)
            self.ball_pos[1] = self.paddle.top - self.ball_radius - 1 # Prevent sticking

        # Blocks
        is_vulnerable = self._is_block_vulnerable()
        for block in self.blocks:
            if block["alive"] and block["rect"].colliderect(ball_rect):
                if is_vulnerable:
                    block["alive"] = False
                    reward += 1.0
                    self.score += 10
                    
                    # Increase ball speed
                    self.ball_speed *= 1.10
                    norm = np.linalg.norm(self.ball_vel)
                    if norm > 0:
                        self.ball_vel = self.ball_vel / norm * self.ball_speed

                    # Create particles
                    self._create_particles(block["rect"].center, block["color"])
                    
                self.ball_vel[1] *= -1 # Bounce regardless of vulnerability
                break # Only handle one block collision per frame

        # --- Update Particles ---
        self._update_particles()

        # --- Termination Conditions ---
        terminated = False
        
        # Win condition
        if not any(b["alive"] for b in self.blocks):
            reward += 100
            terminated = True
            self.game_over = True
        
        # Loss conditions
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            reward -= 100
            terminated = True
            self.game_over = True
            
        if self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _is_block_vulnerable(self):
        oscillation_period = self.FPS  # 1 second
        phase = (self.steps % oscillation_period) / oscillation_period
        return phase < 0.5

    def _get_block_brightness(self):
        oscillation_period = self.FPS
        # Use a sine wave for smooth pulsing
        pulse = (math.sin(self.steps * 2 * math.pi / oscillation_period) + 1) / 2
        return 0.4 + 0.6 * pulse # Varies between 0.4 (dim) and 1.0 (bright)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": np.array(pos, dtype=np.float32), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

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
            "time_remaining": self.time_remaining,
            "blocks_left": sum(1 for b in self.blocks if b["alive"])
        }

    def _render_game(self):
        # Particles (drawn first, to be behind other elements)
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            size = int(max(1, 3 * (p["lifespan"] / 30)))
            color = tuple(int(c * (alpha / 255)) for c in p["color"])
            pygame.draw.rect(self.screen, color, (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # Blocks
        brightness = self._get_block_brightness()
        for block in self.blocks:
            if block["alive"]:
                color = tuple(int(c * brightness) for c in block["color"])
                pygame.draw.rect(self.screen, color, block["rect"], border_radius=3)
                pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), block["rect"], 1, border_radius=3)

        # Paddle with glow
        glow_rect = self.paddle.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_PADDLE_GLOW, 50), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Ball with glow
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_radius = int(self.ball_radius * 2.5)
        
        # Custom glow drawing for performance
        temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (*self.COLOR_BALL_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surface, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, x, y, self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Time
        time_seconds = self.time_remaining // self.FPS
        time_text = self.font_ui.render(f"TIME: {time_seconds}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 5))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not work in a purely headless environment (like the evaluation server),
    # as it requires a display. To run this, you might need to unset SDL_VIDEODRIVER.
    # e.g., in your terminal: `unset SDL_VIDEODRIVER && python your_script_name.py`
    
    # Check if we are in a headless environment and exit if so.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run manual play in headless mode. Exiting.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Setup for manual play window
        pygame.display.set_caption("Block Breaker Environment")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            action = np.array([0, 0, 0])  # Default: no action
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            if keys[pygame.K_r]: # Reset
                obs, info = env.reset()
                total_reward = 0
            if keys[pygame.K_q]:
                running = False

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
                obs, info = env.reset()
                total_reward = 0
            
            clock.tick(GameEnv.FPS)
            
        env.close()