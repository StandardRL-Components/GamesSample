
# Generated: 2025-08-28T05:36:59.451630
# Source Brief: brief_05638.md
# Brief Index: 5638

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hit color-coded balls with your paddle to meet target scores for each color before time runs out or you miss too many."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_WALL = (200, 200, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.BALL_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 120, 255),
            "yellow": (255, 255, 80),
            "purple": (200, 80, 255),
        }
        self.BALL_COLOR_LIST = list(self.BALL_COLORS.values())
        self.BALL_COLOR_NAMES = list(self.BALL_COLORS.keys())
        
        # Game Mechanics
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_Y = self.HEIGHT - 40
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 10
        self.INITIAL_BALL_SPEED = 4
        self.MAX_BALL_SPEED = 10
        self.TARGET_HITS_PER_COLOR = 5
        self.MAX_MISSES = 5

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.paddle = None
        self.ball = None
        self.particles = None
        self.missed_balls = None
        self.color_targets = None
        self.current_ball_speed = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.missed_balls = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.color_targets = {name: self.TARGET_HITS_PER_COLOR for name in self.BALL_COLOR_NAMES}
        
        self.particles = []
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 3=left, 4=right
            
            # Update game logic
            self._update_paddle(movement)
            self._update_ball()
            self._update_particles()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward
            
            self.steps += 1
            
            # Update difficulty
            speed_increase_interval = 10 * self.FPS # every 10 seconds
            if self.steps > 0 and self.steps % speed_increase_interval == 0:
                speed_increase_factor = (self.MAX_BALL_SPEED - self.INITIAL_BALL_SPEED) / (self.MAX_STEPS / speed_increase_interval)
                self.current_ball_speed = min(self.MAX_BALL_SPEED, self.current_ball_speed + speed_increase_factor)

            # Check for termination
            is_win = all(count <= 0 for count in self.color_targets.values())
            is_loss_miss = self.missed_balls >= self.MAX_MISSES
            is_loss_time = self.steps >= self.MAX_STEPS
            
            if is_win:
                self.game_over = True
                reward += 100
                self.score += 100
            elif is_loss_miss or is_loss_time:
                self.game_over = True
                reward -= 100
                self.score -= 100

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_ball(self):
        angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        vel_x = self.current_ball_speed * math.cos(angle)
        vel_y = self.current_ball_speed * math.sin(angle)
        
        color_index = self.np_random.integers(0, len(self.BALL_COLOR_LIST))
        
        self.ball = {
            "pos": pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 4),
            "vel": pygame.math.Vector2(vel_x, vel_y),
            "color_name": self.BALL_COLOR_NAMES[color_index],
            "color_value": self.BALL_COLOR_LIST[color_index],
        }

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(10, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - 10))

    def _update_ball(self):
        self.ball["pos"] += self.ball["vel"]
        
        # Wall collisions
        if self.ball["pos"].x - self.BALL_RADIUS <= 10 or self.ball["pos"].x + self.BALL_RADIUS >= self.WIDTH - 10:
            self.ball["vel"].x *= -1
            # sfx: wall_bounce.wav
        if self.ball["pos"].y - self.BALL_RADIUS <= 10:
            self.ball["vel"].y *= -1
            # sfx: wall_bounce.wav

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball["pos"].x - self.BALL_RADIUS, self.ball["pos"].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Ball-Paddle collision
        if self.paddle.colliderect(ball_rect) and self.ball["vel"].y > 0:
            # sfx: paddle_hit.wav
            self.ball["vel"].y *= -1
            
            # Add spin based on hit location
            offset = (self.ball["pos"].x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball["vel"].x += offset * 2.5
            # Clamp horizontal velocity to prevent extreme angles
            self.ball["vel"].x = max(-self.current_ball_speed, min(self.current_ball_speed, self.ball["vel"].x))

            # Normalize speed
            self.ball["vel"].scale_to_length(self.current_ball_speed)

            # Scoring and rewards
            reward += 0.1
            color_name = self.ball["color_name"]
            if self.color_targets[color_name] > 0:
                reward += 1.0
                self.score += 10
                self.color_targets[color_name] -= 1
            else:
                reward -= 0.5
                self.score -= 5

            self._spawn_particles(self.ball["pos"], self.ball["color_value"])
            self._spawn_ball()
            return reward

        # Ball-Floor miss
        if self.ball["pos"].y > self.HEIGHT:
            # sfx: miss.wav
            self.missed_balls += 1
            self._spawn_ball()
            return 0 # Terminal penalty applied in step()

        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifespan": self.np_random.uniform(15, 30),
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] -= 0.1
        self.particles = [p for p in self.particles if p["lifespan"] > 0 and p["radius"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, pos, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _render_game(self):
        # Playfield border
        pygame.draw.rect(self.screen, self.COLOR_WALL, (5, 5, self.WIDTH - 10, self.HEIGHT - 10), 5, border_radius=5)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30))))
            color_with_alpha = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color_with_alpha
            )

        # Ball
        b_pos = (int(self.ball["pos"].x), int(self.ball["pos"].y))
        pygame.gfxdraw.aacircle(self.screen, b_pos[0], b_pos[1], self.BALL_RADIUS, self.ball["color_value"])
        pygame.gfxdraw.filled_circle(self.screen, b_pos[0], b_pos[1], self.BALL_RADIUS, self.ball["color_value"])
        
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_main, (20, 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Misses
        miss_text = "MISSES: " + "X " * self.missed_balls + "_ " * (self.MAX_MISSES - self.missed_balls)
        misses_surf = self.font_main.render(miss_text, True, self.COLOR_TEXT)
        self.screen.blit(misses_surf, (self.WIDTH - misses_surf.get_width() - 20, 20))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"{time_left:.1f}"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, ((self.WIDTH - timer_surf.get_width()) / 2, self.HEIGHT - 35))

        # Color Targets
        target_box_size = 30
        total_width = len(self.BALL_COLORS) * (target_box_size + 10) - 10
        start_x = (self.WIDTH - total_width) / 2
        
        for i, (name, color) in enumerate(self.BALL_COLORS.items()):
            x = start_x + i * (target_box_size + 10)
            y = 20
            rect = pygame.Rect(x, y, target_box_size, target_box_size)
            
            count = self.color_targets[name]
            alpha = 100 if count <= 0 else 255
            
            # Draw box with transparency
            s = pygame.Surface((target_box_size, target_box_size), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (x, y))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=3)
            
            # Draw count
            count_text = str(count)
            count_surf = self.font_small.render(count_text, True, self.COLOR_TEXT)
            text_pos = (x + (target_box_size - count_surf.get_width()) / 2, y + target_box_size + 5)
            self._render_text(count_text, self.font_small, text_pos, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Game Over Message
        if self.game_over:
            is_win = all(count <= 0 for count in self.color_targets.values())
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = self.BALL_COLORS["green"] if is_win else self.BALL_COLORS["red"]
            
            # Create a semi-transparent overlay
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            font_large = pygame.font.SysFont("Consolas", 72, bold=True)
            self._render_text(msg, font_large, ((self.WIDTH - font_large.size(msg)[0])/2, (self.HEIGHT - font_large.size(msg)[1])/2), color, self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_balls": self.missed_balls,
            "targets_remaining": self.color_targets,
            "time_remaining_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
        
    def close(self):
        pygame.quit()

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: Gymnasium environments are not typically run this way for training,
    # but it's useful for testing and visualization.
    
    # To run, you need to set the video driver for pygame.
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "directfb", "fbcon", etc. for Linux
                                         # Use "windows" for Windows
                                         # Use "quartz" for MacOS
                                         
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Color Pong")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are unused

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Check for game over from the environment
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before closing
            pygame.time.wait(3000)

    env.close()