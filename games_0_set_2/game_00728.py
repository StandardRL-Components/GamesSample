
# Generated: 2025-08-27T14:35:01.774638
# Source Brief: brief_00728.md
# Brief Index: 728

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade survival game. Bounce multiple balls with your paddle and survive for 20 seconds. "
        "The balls get faster over time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_DURATION_SECONDS = 20
        self.MAX_STEPS = self.MAX_DURATION_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BALL_COLORS = [(255, 80, 80), (80, 255, 80), (80, 150, 255)]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_Y = self.SCREEN_HEIGHT - 40
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 3.0
        self.SPEED_INCREASE_INTERVAL = 200
        self.SPEED_INCREASE_AMOUNT = 0.5
        self.NUM_BALLS = 3

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 36)
            self.font_small = pygame.font.SysFont("monospace", 24)

        # Initialize state variables
        self.paddle = None
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle state
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Initialize balls
        self.balls = []
        for i in range(self.NUM_BALLS):
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
            vel = pygame.Vector2(math.cos(angle), -math.sin(angle)) * self.INITIAL_BALL_SPEED
            self.balls.append({
                "pos": pygame.Vector2(
                    self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS),
                    self.np_random.uniform(self.SCREEN_HEIGHT * 0.1, self.SCREEN_HEIGHT * 0.4)
                ),
                "vel": vel,
                "speed": self.INITIAL_BALL_SPEED,
                "color": self.BALL_COLORS[i % len(self.BALL_COLORS)]
            })

        # Reset game state
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        
        # --- 1. Handle Player Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # --- 2. Update Game Logic ---
        bounce_reward = self._update_balls()
        self._update_particles()
        
        # --- 3. Calculate Reward & Termination ---
        reward = 0.1  # Survival reward per frame
        reward += bounce_reward
        self.score += reward

        terminated = self.game_over

        if self.game_over:
            reward = -10.0 # Penalty for missing a ball
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward += 100.0 # Bonus for survival
            self.score += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_balls(self):
        bounce_reward = 0
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            for ball in self.balls:
                ball["speed"] += self.SPEED_INCREASE_AMOUNT

        for ball in self.balls:
            ball["pos"] += ball["vel"]
            
            # Wall collisions
            if ball["pos"].x - self.BALL_RADIUS <= 0 or ball["pos"].x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
                ball["vel"].x *= -1
                ball["pos"].x = max(self.BALL_RADIUS, min(ball["pos"].x, self.SCREEN_WIDTH - self.BALL_RADIUS))
                # sfx: wall_bounce.wav
            if ball["pos"].y - self.BALL_RADIUS <= 0:
                ball["vel"].y *= -1
                ball["pos"].y = self.BALL_RADIUS
                # sfx: wall_bounce.wav

            # Paddle collision
            if ball["vel"].y > 0 and self.paddle.collidepoint(ball["pos"].x, ball["pos"].y + self.BALL_RADIUS):
                ball["pos"].y = self.paddle.top - self.BALL_RADIUS
                ball["vel"].y *= -1
                
                # Add spin based on where it hits the paddle
                hit_pos_norm = (ball["pos"].x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                ball["vel"].x += hit_pos_norm * 2.0
                
                # Maintain speed
                if ball["vel"].length() > 0:
                    ball["vel"].scale_to_length(ball["speed"])

                bounce_reward += 1.0
                self._create_particles(ball["pos"], ball["color"])
                # sfx: paddle_hit.wav

            # Out of bounds (miss)
            if ball["pos"].y - self.BALL_RADIUS > self.SCREEN_HEIGHT:
                self.game_over = True
                # sfx: game_over.wav

        return bounce_reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] -= 0.1
        self.particles = [p for p in self.particles if p["lifespan"] > 0 and p["radius"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_particles()
        self._render_balls()
        self._render_paddle()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 64):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_paddle(self):
        # Draw a subtle glow/trail
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

    def _render_balls(self):
        for ball in self.balls:
            x, y = int(ball["pos"].x), int(ball["pos"].y)
            radius = int(self.BALL_RADIUS)
            color = ball["color"]
            
            # Anti-aliased circle drawing for high quality
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            
            # Add a white highlight for a 3D effect
            highlight_x = int(x - radius * 0.3)
            highlight_y = int(y - radius * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, int(radius * 0.3), (255, 255, 255, 150))

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p["pos"].x), int(p["pos"].y)
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["lifespan"] / 30))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (x - radius, y - radius))

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{time_left:.1f}s", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win message
        if terminated := (self.game_over or self.steps >= self.MAX_STEPS):
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "SURVIVED!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            msg_surf = self.font_large.render(message, True, self.COLOR_PADDLE)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            final_score_surf = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get an observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        
        # Test info from reset
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            # The other actions are unused in this game
            space_held = 0
            shift_held = 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
            # If terminated, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()