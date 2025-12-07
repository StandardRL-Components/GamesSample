
# Generated: 2025-08-28T02:54:26.106829
# Source Brief: brief_01851.md
# Brief Index: 1851

        
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

    user_guide = (
        "Controls: ↑ to move the paddle up, ↓ to move down. "
        "Reflect the ball to score points."
    )

    game_description = (
        "A cyberpunk-themed arcade game. Reflect the energy ball with your paddle to score points. "
        "Reach 15 points to win. Miss the ball 3 times and you lose. "
        "Risky edge-shots and power-ups grant bonus points."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.GRID_ROWS = 30
        self.GRID_COLS = 40
        self.WIN_SCORE = 15
        self.MAX_LIVES = 3
        self.MAX_STEPS = 2000

        self.PADDLE_WIDTH = 12
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 7
        self.MAX_BALL_SPEED = 15

        # --- Colors (Neon Cyberpunk) ---
        self.COLOR_BG = (10, 10, 30)
        self.COLOR_GRID = (30, 50, 90)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_BALL = (50, 255, 150)
        self.COLOR_POWERUP = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_DANGER = (255, 50, 100)
        
        # --- Fonts ---
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State variables will be initialized in reset() ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.powerups = []
        self.active_powerup = {"type": None, "timer": 0}
        
        # --- Initialize state ---
        self.reset()
        
        # --- Validate implementation ---
        # self.validate_implementation() # Commented out for submission, but used for development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.particles = []
        self.powerups = []
        self.active_powerup = {"type": None, "timer": 0}

        # Reset paddle
        paddle_height = self.screen_height / 5
        self.paddle = pygame.Rect(
            self.screen_width - self.PADDLE_WIDTH * 2,
            self.screen_height / 2 - paddle_height / 2,
            self.PADDLE_WIDTH,
            paddle_height
        )

        # Reset ball
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.screen_width / 2, self.screen_height / 2)
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if self.np_random.random() < 0.5:
            angle += math.pi # Send ball to the left
        
        # Ensure ball doesn't start moving towards player
        angle = self.np_random.choice([-math.pi/4, math.pi/4])

        self.ball_vel = pygame.Vector2(
            math.cos(angle) * self.INITIAL_BALL_SPEED,
            math.sin(angle) * self.INITIAL_BALL_SPEED
        )

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self._update_paddle(movement)
        
        reward, events = self._update_game_state()
        
        self._update_particles()
        self._handle_powerups(events)

        self.score += events.get("score_gain", 0)
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                # Sound: Win Jingle
            elif self.lives <= 0:
                reward -= 100
                # Sound: Game Over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED

        self.paddle.y = np.clip(self.paddle.y, 0, self.screen_height - self.paddle.height)

    def _update_game_state(self):
        reward = 0
        events = {}

        # Update ball position
        self.ball_pos += self.ball_vel

        # Ball collision with top/bottom walls
        if self.ball_pos.y - self.BALL_RADIUS <= 0 or self.ball_pos.y + self.BALL_RADIUS >= self.screen_height:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.screen_height - self.BALL_RADIUS)
            self._spawn_particles(self.ball_pos, 5, self.COLOR_BALL)
            # Sound: Wall Bounce

        # Ball collision with left wall
        if self.ball_pos.x - self.BALL_RADIUS <= 0:
            self.ball_vel.x *= -1
            self.ball_pos.x = self.BALL_RADIUS
            self._spawn_particles(self.ball_pos, 5, self.COLOR_BALL)
            # Sound: Wall Bounce

        # Ball misses paddle (right wall)
        if self.ball_pos.x + self.BALL_RADIUS >= self.screen_width:
            self.lives -= 1
            self._spawn_particles(self.ball_pos, 30, self.COLOR_DANGER, speed_mult=2.0)
            # Sound: Life Lost
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True

        # Ball collision with paddle
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.ball_vel.x > 0 and self.paddle.colliderect(ball_rect):
            self.ball_vel.x *= -1.05 # Reverse and slightly speed up
            self.ball_pos.x = self.paddle.left - self.BALL_RADIUS

            # Calculate bounce angle based on impact point
            offset = (self.paddle.centery - self.ball_pos.y) / (self.paddle.height / 2)
            offset = np.clip(offset, -1, 1)
            self.ball_vel.y -= offset * 5
            
            # Clamp ball speed
            speed = self.ball_vel.length()
            if speed > self.MAX_BALL_SPEED:
                self.ball_vel.scale_to_length(self.MAX_BALL_SPEED)

            # Calculate reward
            reward += 0.1  # Base reward for hitting
            score_multiplier = 2 if self.active_powerup["type"] == "double_points" else 1
            
            if abs(offset) > 0.7: # Risky edge hit
                reward += 2.0
                events["score_gain"] = 1 * score_multiplier
                self._spawn_particles(self.ball_pos, 20, self.COLOR_POWERUP)
                # Sound: Risky Hit
            else:
                if abs(offset) < 0.2: # Safe center hit
                    reward -= 0.02
                self._spawn_particles(self.ball_pos, 10, self.COLOR_PADDLE)
                # Sound: Paddle Hit
        
        return reward, events
    
    def _handle_powerups(self, events):
        # Spawn new powerup
        if self.steps > 0 and self.steps % 450 == 0 and not self.powerups and self.active_powerup["timer"] <= 0:
            p_type = self.np_random.choice(["large_paddle", "double_points"])
            self.powerups.append({
                "pos": pygame.Vector2(self.np_random.integers(100, self.screen_width - 200), self.np_random.integers(50, self.screen_height - 50)),
                "type": p_type,
                "life": 300 # 10 seconds at 30fps
            })
        
        # Update and check collection
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        for p in self.powerups[:]:
            p["life"] -= 1
            p_rect = pygame.Rect(p["pos"].x - 15, p["pos"].y - 15, 30, 30)
            if p_rect.colliderect(ball_rect):
                self.active_powerup = {"type": p["type"], "timer": 450} # 15 seconds
                if p["type"] == "large_paddle":
                    self.paddle.height *= 1.5
                    self.paddle.y -= self.paddle.height * 0.25
                events["score_gain"] = events.get("score_gain", 0) + 1
                self._spawn_particles(p["pos"], 40, self.COLOR_POWERUP, speed_mult=1.5)
                # Sound: Powerup Collect
                self.powerups.remove(p)
            elif p["life"] <= 0:
                self.powerups.remove(p)

        # Update active powerup timer
        if self.active_powerup["timer"] > 0:
            self.active_powerup["timer"] -= 1
            if self.active_powerup["timer"] == 0:
                if self.active_powerup["type"] == "large_paddle":
                    self.paddle.y += self.paddle.height * 0.25
                    self.paddle.height /= 1.5
                self.active_powerup["type"] = None
                # Sound: Powerup End

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })
            
    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        cell_w = self.screen_width / self.GRID_COLS
        cell_h = self.screen_height / self.GRID_ROWS
        for i in range(self.GRID_COLS + 1):
            x = int(i * cell_w)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height), 1)
        for i in range(self.GRID_ROWS + 1):
            y = int(i * cell_h)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y), 1)

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)

        # Powerups
        for p in self.powerups:
            flash = abs(math.sin(self.steps * 0.2))
            alpha = int(150 + 100 * flash)
            self._draw_glow_circle(self.screen, p["pos"], 15, self.COLOR_POWERUP, alpha)
            
        # Paddle
        self._draw_glow_rect(self.screen, self.paddle, self.COLOR_PADDLE, 10)
        
        # Ball
        self._draw_glow_circle(self.screen, self.ball_pos, self.BALL_RADIUS, self.COLOR_BALL, 255)
    
    def _draw_glow_rect(self, surface, rect, color, glow_size):
        glow_color = (*color, 50)
        glow_rect = rect.inflate(glow_size, glow_size)
        pygame.draw.rect(surface, glow_color, glow_rect, border_radius=8)
        pygame.draw.rect(surface, color, rect, border_radius=5)
        
    def _draw_glow_circle(self, surface, center, radius, color, alpha):
        center_int = (int(center.x), int(center.y))
        for i in range(4, 0, -1):
            glow_alpha = int(alpha * (0.1 * (5-i)))
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1],
                                         int(radius + i * 2), (*color, glow_alpha))
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Lives
        for i in range(self.MAX_LIVES):
            pos = (self.screen_width - 30 - i * 25, 25)
            color = self.COLOR_BALL if i < self.lives else self.COLOR_DANGER
            self._draw_glow_circle(self.screen, pygame.Vector2(*pos), 8, color, 180 if i < self.lives else 255)

        # Active Powerup
        if self.active_powerup["timer"] > 0:
            name = self.active_powerup["type"].replace("_", " ").title()
            timer_secs = self.active_powerup["timer"] / 30
            text = f"{name} ({timer_secs:.1f}s)"
            powerup_text = self.font_small.render(text, True, self.COLOR_POWERUP)
            text_rect = powerup_text.get_rect(centerx=self.screen_width / 2, y=15)
            self.screen.blit(powerup_text, text_rect)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_BALL if self.score >= self.WIN_SCORE else self.COLOR_DANGER
            end_text = self.font_main.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Neon Reflect")
    
    running = True
    total_reward = 0
    
    while running:
        # Player controls
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
            
        action = [movement, 0, 0] # Other actions are not used in this game
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()