
# Generated: 2025-08-28T00:52:20.715875
# Source Brief: brief_03928.md
# Brief Index: 3928

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Short, user-facing control string
    user_guide = "Controls: ←→ to move the paddle."

    # Short, user-facing description of the game
    game_description = "Navigate a bouncing ball through a twisting neon tunnel. Maximize your score with risky edge-of-the-paddle bounces."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 20
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.MAX_LIVES = 5

        # EXACT spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_BG_GRADIENT = [(2, 0, 18), (5, 0, 30), (10, 0, 40)]
        self.COLOR_WALL = (50, 50, 150)
        self.COLOR_PADDLE = (0, 255, 255)  # Cyan
        self.COLOR_PADDLE_GLOW = (0, 150, 150)
        self.COLOR_BALL = (255, 255, 0)    # Yellow
        self.COLOR_BALL_GLOW = (150, 150, 0)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WARN = (255, 0, 0)

        # Fonts
        self.font_ui = pygame.font.Font(None, 36)
        self.font_warn = pygame.font.Font(None, 24)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.initial_ball_speed = 3.0
        self.ball_speed_multiplier = 1.0
        self.ball_radius = 8
        self.paddle_x = 0
        self.paddle_width = 100
        self.paddle_height = 10
        self.paddle_speed = 8
        self.particles = []
        self.tunnel = []
        self.tunnel_length = 8000
        self.tunnel_segment_length = 20
        self.current_segment_index = 0
        self.camera_y = 0.0
        self.warn_flash_timer = 0

        # This will be properly initialized in reset()
        self.np_random = None

        self.validate_implementation()

    def _generate_tunnel(self):
        self.tunnel = []
        center_x = self.WIDTH / 2
        width = self.WIDTH * 0.8
        y = 0
        
        # Use a combination of sine waves for natural-looking curves
        freq1, amp1 = self.np_random.uniform(0.001, 0.002), self.np_random.uniform(100, 150)
        freq2, amp2 = self.np_random.uniform(0.005, 0.008), self.np_random.uniform(30, 50)
        
        while y < self.tunnel_length:
            offset = math.sin(y * freq1) * amp1 + math.sin(y * freq2) * amp2
            current_center = center_x + offset
            
            # Tunnel gets narrower over time
            current_width = width * max(0.2, 1.0 - (y / self.tunnel_length) * 0.9)
            
            left_x = current_center - current_width / 2
            right_x = current_center + current_width / 2
            self.tunnel.append({"y": y, "left": left_x, "right": right_x})
            y += self.tunnel_segment_length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_tunnel()

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.ball_speed_multiplier = 1.0

        self.paddle_x = self.WIDTH / 2
        self.camera_y = 0
        self.current_segment_index = 0
        self.particles = []
        self.warn_flash_timer = 0
        
        self._reset_ball(0)

        return self._get_observation(), self._get_info()

    def _reset_ball(self, segment_index):
        start_segment = self.tunnel[segment_index]
        self.ball_pos = np.array([
            (start_segment["left"] + start_segment["right"]) / 2,
            float(start_segment["y"] + 50)
        ])
        
        angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6)
        initial_speed = self.initial_ball_speed * self.ball_speed_multiplier
        self.ball_vel = np.array([math.cos(angle) * initial_speed, math.sin(angle) * initial_speed])
        if self.np_random.random() < 0.5:
            self.ball_vel[0] *= -1

    def step(self, action):
        movement = action[0]
        reward = 0.01  # Small reward for surviving
        terminated = False

        # 1. Update Paddle
        if movement == 3:  # Left
            self.paddle_x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle_x += self.paddle_speed
        
        # Clamp paddle to be inside the current visible tunnel segment
        # This provides better feedback than clamping to screen edges
        paddle_world_y = self.camera_y + self.HEIGHT - 30
        paddle_segment_idx = self._get_segment_index_at_y(paddle_world_y)
        if 0 <= paddle_segment_idx < len(self.tunnel):
            segment = self.tunnel[paddle_segment_idx]
            self.paddle_x = np.clip(self.paddle_x, segment["left"] + self.paddle_width / 2, segment["right"] - self.paddle_width / 2)
        else:
             self.paddle_x = np.clip(self.paddle_x, self.paddle_width/2, self.WIDTH - self.paddle_width/2)

        # 2. Update Ball
        self.ball_pos += self.ball_vel

        # 3. Handle Collisions & Game Logic
        # Ball vs Paddle
        paddle_top_y = self.camera_y + self.HEIGHT - 30
        paddle_left_x = self.paddle_x - self.paddle_width / 2
        paddle_right_x = self.paddle_x + self.paddle_width / 2

        if (self.ball_vel[1] > 0 and 
            paddle_top_y < self.ball_pos[1] + self.ball_radius < paddle_top_y + self.paddle_height and
            paddle_left_x < self.ball_pos[0] < paddle_right_x):
            
            # Reflect velocity
            self.ball_vel[1] *= -1
            self.ball_pos[1] = paddle_top_y - self.ball_radius # Prevent sticking
            
            # Influence horizontal velocity based on impact point
            impact_norm = (self.ball_pos[0] - self.paddle_x) / (self.paddle_width / 2)
            self.ball_vel[0] += impact_norm * 2.0
            
            # Normalize speed
            speed = np.linalg.norm(self.ball_vel)
            target_speed = self.initial_ball_speed * self.ball_speed_multiplier
            if speed > 0:
                self.ball_vel = self.ball_vel * (target_speed / speed)

            # Calculate reward
            if abs(impact_norm) > 0.5: # Risky bounce
                reward += 0.5
                self.score += 50
                self._create_particles(self.ball_pos, 20, self.COLOR_BALL)
                # sfx: high_pitched_bounce
            else: # Safe bounce
                reward -= 0.2
                self.score += 10
                self._create_particles(self.ball_pos, 5, self.COLOR_PADDLE)
                # sfx: normal_bounce

        # Ball vs Walls
        ball_segment_idx = self._get_segment_index_at_y(self.ball_pos[1])
        if 0 <= ball_segment_idx < len(self.tunnel):
            segment = self.tunnel[ball_segment_idx]
            if self.ball_pos[0] - self.ball_radius < segment["left"]:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = segment["left"] + self.ball_radius
                self._create_particles(np.array([segment["left"], self.ball_pos[1]]), 3)
                # sfx: wall_thud
            elif self.ball_pos[0] + self.ball_radius > segment["right"]:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = segment["right"] - self.ball_radius
                self._create_particles(np.array([segment["right"], self.ball_pos[1]]), 3)
                # sfx: wall_thud

        # 4. Check Progress & Update Difficulty
        if ball_segment_idx > self.current_segment_index:
            reward += 1.0 * (ball_segment_idx - self.current_segment_index)
            self.score += 100 * (ball_segment_idx - self.current_segment_index)
            self.current_segment_index = ball_segment_idx
        
        if self.steps > 0 and self.steps % 100 == 0:
            self.ball_speed_multiplier += 0.05

        # 5. Check Termination Conditions
        # Missed ball
        if self.ball_pos[1] > self.camera_y + self.HEIGHT:
            self.lives -= 1
            reward -= 10.0
            self.warn_flash_timer = 30 # Flash red for 0.5s
            # sfx: lose_life
            if self.lives <= 0:
                terminated = True
                reward -= 100.0 # Big penalty for game over
                # sfx: game_over
            else:
                self._reset_ball(self.current_segment_index)
        
        # Win condition
        if self.ball_pos[1] >= self.tunnel_length:
            terminated = True
            reward += 100.0
            self.score += 10000
            # sfx: win_fanfare

        # Time limit
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            # No extra penalty, loss of potential future rewards is enough

        # 6. Update Particles & Camera
        self._update_particles()
        target_camera_y = self.ball_pos[1] - self.HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.08
        self.camera_y = max(0, self.camera_y)

        if self.warn_flash_timer > 0:
            self.warn_flash_timer -= 1
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_segment_index_at_y(self, y_pos):
        return int(max(0, y_pos) / self.tunnel_segment_length)

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": lifetime, "max_life": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Draw background gradient
        for i, color in enumerate(self.COLOR_BG_GRADIENT):
            pygame.draw.rect(self.screen, color, (0, i * self.HEIGHT/3, self.WIDTH, self.HEIGHT/3 + 1))
        
        # Draw tunnel
        start_idx = self._get_segment_index_at_y(self.camera_y)
        end_idx = self._get_segment_index_at_y(self.camera_y + self.HEIGHT) + 2
        
        for i in range(start_idx, min(end_idx, len(self.tunnel) - 1)):
            p1 = self.tunnel[i]
            p2 = self.tunnel[i+1]
            
            # Transform to screen coordinates
            p1_ly, p1_ry = p1["y"] - self.camera_y, p1["y"] - self.camera_y
            p2_ly, p2_ry = p2["y"] - self.camera_y, p2["y"] - self.camera_y
            
            # Draw filled polygons for walls for a solid look
            pygame.draw.polygon(self.screen, self.COLOR_WALL, [
                (0, p1_ly), (p1["left"], p1_ly), 
                (p2["left"], p2_ly), (0, p2_ly)
            ])
            pygame.draw.polygon(self.screen, self.COLOR_WALL, [
                (self.WIDTH, p1_ry), (p1["right"], p1_ry), 
                (p2["right"], p2_ry), (self.WIDTH, p2_ry)
            ])

        # Draw particles
        for p in self.particles:
            screen_pos = p["pos"] - np.array([0, self.camera_y])
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"]
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), 2, (*color, alpha))

        # Draw paddle
        paddle_screen_y = self.HEIGHT - 30
        paddle_rect = pygame.Rect(0, 0, self.paddle_width, self.paddle_height)
        paddle_rect.center = (self.paddle_x, paddle_screen_y)
        pygame.gfxdraw.box(self.screen, paddle_rect, (*self.COLOR_PADDLE_GLOW, 100))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw ball
        ball_screen_pos = self.ball_pos - np.array([0, self.camera_y])
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos[0]), int(ball_screen_pos[1]), self.ball_radius + 4, (*self.COLOR_BALL_GLOW, 100))
        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos[0]), int(ball_screen_pos[1]), self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos[0]), int(ball_screen_pos[1]), self.ball_radius, self.COLOR_BALL)

        # Draw UI
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.TIME_LIMIT_SECONDS - self.steps / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, self.HEIGHT - lives_text.get_height() - 10))
        
        # Draw win/loss message
        if self.ball_pos[1] >= self.tunnel_length:
            win_text = self.font_ui.render("TUNNEL COMPLETE!", True, self.COLOR_PADDLE)
            self.screen.blit(win_text, (self.WIDTH/2 - win_text.get_width()/2, self.HEIGHT/2 - win_text.get_height()/2))
        elif self.lives <= 0:
            lose_text = self.font_ui.render("GAME OVER", True, self.COLOR_WARN)
            self.screen.blit(lose_text, (self.WIDTH/2 - lose_text.get_width()/2, self.HEIGHT/2 - lose_text.get_height()/2))

        # Warning flash on life loss
        if self.warn_flash_timer > 0:
            flash_alpha = int(100 * (self.warn_flash_timer / 30))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_WARN, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "progress": self.ball_pos[1] / self.tunnel_length
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not terminated:
        # Map pygame keys to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()