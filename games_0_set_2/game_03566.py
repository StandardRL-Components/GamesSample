
# Generated: 2025-08-27T23:45:33.440082
# Source Brief: brief_03566.md
# Brief Index: 3566

        
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


# Helper classes for game objects
class Obstacle:
    def __init__(self, screen_height, rng):
        self.w = rng.integers(30, 60)
        self.h = rng.integers(30, 60)
        self.x = 640 + self.w
        self.y = rng.integers(80, screen_height - 80 - self.h)
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        self.color = (139, 69, 19)  # SaddleBrown
        self.outline_color = (101, 67, 33) # Darker brown

    def update(self, scroll_speed):
        self.x -= scroll_speed
        self.rect.x = int(self.x)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=8)
        pygame.draw.rect(surface, self.outline_color, self.rect, width=3, border_radius=8)

class Particle:
    def __init__(self, x, y, rng):
        self.x = x
        self.y = y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(2, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = rng.integers(15, 30)
        self.color = random.choice([(255, 0, 0), (255, 69, 0), (255, 165, 0)]) # Red/Orange
        self.size = rng.integers(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.vx *= 0.95
        self.vy *= 0.95
        self.size = max(0, self.size - 0.2)

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←/→ to steer up and down."
    )

    game_description = (
        "Guide your speedy snail to the finish line, dodging rocks and racing against the clock in this vibrant side-scrolling arcade racer."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (124, 252, 0)  # LawnGreen
    COLOR_TRACK = (127, 181, 76) # Darker Green
    COLOR_SNAIL_BODY = (255, 105, 180) # HotPink
    COLOR_SNAIL_SHELL = (255, 69, 0) # OrangeRed
    COLOR_SNAIL_SHELL_OUTLINE = (210, 50, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_FINISH_LINE_1 = (255, 255, 0) # Yellow
    COLOR_FINISH_LINE_2 = (255, 255, 255) # White

    # Game parameters
    FINISH_LINE_DISTANCE = 25000
    TIME_LIMIT_SECONDS = 60
    MAX_HITS = 5
    TIME_LIMIT_STEPS = TIME_LIMIT_SECONDS * FPS

    # Snail physics
    SNAIL_X_POS = 100
    SNAIL_ACCEL = 0.2
    SNAIL_BRAKE = 0.4
    SNAIL_SPEED_FRICTION = 0.99
    SNAIL_TURN_ACCEL = 1.5
    SNAIL_Y_FRICTION = 0.85
    MAX_SPEED = 20.0
    MIN_SPEED = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        self.snail_body_surf = pygame.Surface((60, 30), pygame.SRCALPHA)
        pygame.draw.ellipse(self.snail_body_surf, self.COLOR_SNAIL_BODY, (0, 0, 60, 25))
        pygame.draw.circle(self.snail_body_surf, (255,255,255), (48, 8), 5)
        pygame.draw.circle(self.snail_body_surf, (0,0,0), (50, 8), 3)

        self.game_over = True # Ensure reset is called first
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.hits = 0
        self.distance_traveled = 0

        # Snail state
        self.snail_y = self.SCREEN_HEIGHT / 2
        self.snail_vy = 0
        self.speed = self.MIN_SPEED
        self.target_speed = self.MIN_SPEED

        # Entities
        self.obstacles = []
        self.particles = []
        self.speed_lines = []

        # Difficulty
        self.obstacle_spawn_timer = 0
        self.base_spawn_prob = 0.5 / self.FPS # 0.5 per second
        self.max_spawn_prob = 1.0 / self.FPS
        self.current_spawn_prob = self.base_spawn_prob

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Base reward for surviving a step

        movement, _, _ = action
        self._handle_input(movement)
        self._update_game_state()

        # --- Collision Detection ---
        snail_rect = self._get_snail_rect()
        for obstacle in self.obstacles:
            if snail_rect.colliderect(obstacle.rect):
                self.hits += 1
                reward -= 5
                self.obstacles.remove(obstacle)
                # SFX: Crash sound
                for _ in range(20):
                    self.particles.append(Particle(snail_rect.centerx, snail_rect.centery, self.rng))
                if self.hits >= self.MAX_HITS:
                    self.game_over = True
                    reward -= 100 # Penalty for losing by crashing
                break

        self.steps += 1
        
        # --- Termination and Terminal Rewards ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.won:
                time_taken = self.steps / self.FPS
                time_bonus = max(0, self.TIME_LIMIT_SECONDS - time_taken)
                win_reward = 100 * (time_bonus / self.TIME_LIMIT_SECONDS)
                reward += win_reward
                # SFX: Victory fanfare
            else: # Timed out
                reward -= 50 # Penalty for timing out
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Action 0: no-op
        # Action 1 (up): Accelerate
        if movement == 1:
            self.target_speed += self.SNAIL_ACCEL
        # Action 2 (down): Decelerate
        elif movement == 2:
            self.target_speed -= self.SNAIL_BRAKE
        
        self.target_speed = np.clip(self.target_speed, self.MIN_SPEED, self.MAX_SPEED)

        # Action 3 (left): Steer up
        if movement == 3:
            self.snail_vy -= self.SNAIL_TURN_ACCEL
        # Action 4 (right): Steer down
        elif movement == 4:
            self.snail_vy += self.SNAIL_TURN_ACCEL

    def _update_game_state(self):
        # Update snail speed (smoothly)
        self.speed += (self.target_speed - self.speed) * 0.1
        self.speed *= self.SNAIL_SPEED_FRICTION
        self.distance_traveled += self.speed

        # Update snail position (smoothly)
        self.snail_y += self.snail_vy
        self.snail_vy *= self.SNAIL_Y_FRICTION
        
        # Snail bounds
        track_top, track_bottom = 60, self.SCREEN_HEIGHT - 60
        snail_height = 40
        self.snail_y = np.clip(self.snail_y, track_top + snail_height/2, track_bottom - snail_height/2)

        # Update obstacles
        for o in self.obstacles:
            o.update(self.speed)
        self.obstacles = [o for o in self.obstacles if o.rect.right > 0]
        
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

        # Update speed lines
        if self.speed > self.MIN_SPEED + 1:
            line_start_y = self.snail_y - 5 + self.rng.uniform(0, 10)
            self.speed_lines.append([self.SNAIL_X_POS, line_start_y, self.speed / self.MAX_SPEED])
        for line in self.speed_lines:
            line[0] -= self.speed * 1.5 # Move lines faster than world
        self.speed_lines = [line for line in self.speed_lines if line[0] > 0]

        # Spawn new obstacles
        self.current_spawn_prob = min(self.max_spawn_prob, self.base_spawn_prob + (self.steps / self.TIME_LIMIT_STEPS) * (self.max_spawn_prob - self.base_spawn_prob))
        if self.rng.random() < self.current_spawn_prob:
            self.obstacles.append(Obstacle(self.SCREEN_HEIGHT, self.rng))

    def _get_snail_rect(self):
        return pygame.Rect(self.SNAIL_X_POS - 20, int(self.snail_y) - 20, 50, 40)

    def _check_termination(self):
        if self.hits >= self.MAX_HITS:
            return True
        if self.steps >= self.TIME_LIMIT_STEPS:
            return True
        if self.distance_traveled >= self.FINISH_LINE_DISTANCE:
            self.won = True
            return True
        return False
        
    def _get_observation(self):
        self._render_background()
        self._render_finish_line()
        self._render_obstacles()
        self._render_speed_lines()
        self._render_snail()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        
        # Parallax hills
        for i in range(3):
            scroll_factor = 0.2 + i * 0.1
            offset = -((self.distance_traveled * scroll_factor) % self.SCREEN_WIDTH)
            color = (107 - i*10, 161 - i*10, 56 - i*10)
            pygame.draw.rect(self.screen, color, (offset, 20 + i*20, self.SCREEN_WIDTH, 100))
            pygame.draw.rect(self.screen, color, (offset + self.SCREEN_WIDTH, 20 + i*20, self.SCREEN_WIDTH, 100))

        # Main track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, 60, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 120))
        
        # Track lines
        for i in range(15):
            line_x = (i * 50 - (self.distance_traveled % 50))
            pygame.draw.line(self.screen, (255,255,255,50), (line_x, 60), (line_x, self.SCREEN_HEIGHT-60), 2)


    def _render_obstacles(self):
        for o in self.obstacles:
            o.draw(self.screen)
            
    def _render_finish_line(self):
        finish_x = self.FINISH_LINE_DISTANCE - self.distance_traveled + self.SNAIL_X_POS
        if 0 < finish_x < self.SCREEN_WIDTH + 50:
            track_top, track_bottom = 60, self.SCREEN_HEIGHT - 60
            check_size = 20
            for y in range(track_top, track_bottom, check_size):
                for x_offset in range(0, 50, check_size * 2):
                    color_1 = self.COLOR_FINISH_LINE_1 if (y // check_size) % 2 == 0 else self.COLOR_FINISH_LINE_2
                    color_2 = self.COLOR_FINISH_LINE_2 if (y // check_size) % 2 == 0 else self.COLOR_FINISH_LINE_1
                    pygame.draw.rect(self.screen, color_1, (int(finish_x + x_offset), y, check_size, check_size))
                    pygame.draw.rect(self.screen, color_2, (int(finish_x + x_offset + check_size), y, check_size, check_size))
            pygame.draw.line(self.screen, (0,0,0), (int(finish_x), track_top), (int(finish_x), track_bottom), 3)


    def _render_snail(self):
        x, y = self.SNAIL_X_POS, int(self.snail_y)
        
        # Draw body
        body_rotated = pygame.transform.rotate(self.snail_body_surf, -self.snail_vy * 2)
        body_rect = body_rotated.get_rect(center=(x, y + 5))
        self.screen.blit(body_rotated, body_rect)
        
        # Draw shell using gfxdraw for anti-aliasing
        shell_x, shell_y = x - 15, y - 5
        shell_radius = 20
        
        # Spiral effect
        for i in range(8):
            rad = shell_radius * (1 - i/10.0)
            angle_offset = self.distance_traveled * 0.02
            pygame.gfxdraw.arc(self.screen, shell_x, shell_y, int(rad), int(200 + i*20 + angle_offset*20), int(340 + i*20 + angle_offset*20), self.COLOR_SNAIL_SHELL_OUTLINE)

        pygame.gfxdraw.aacircle(self.screen, shell_x, shell_y, shell_radius, self.COLOR_SNAIL_SHELL_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, shell_x, shell_y, shell_radius, self.COLOR_SNAIL_SHELL)
        
    def _render_speed_lines(self):
        for line in self.speed_lines:
            x, y, intensity = line
            alpha = int(200 * intensity * (x / self.SNAIL_X_POS))
            color = (255, 255, 255, alpha)
            length = self.speed * 1.5
            start_pos = (int(x), int(y))
            end_pos = (int(x - length), int(y))
            if start_pos[0] > end_pos[0]: # Ensure line has positive length
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, x, y, color=self.COLOR_TEXT):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (x + 2, y + 2))
            self.screen.blit(content, (x, y))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = f"TIME: {minutes:02}:{seconds:02}"
        draw_text(timer_text, self.font_small, 10, 10)

        # Hits
        hits_text = f"HITS: {self.hits}/{self.MAX_HITS}"
        draw_text(hits_text, self.font_small, self.SCREEN_WIDTH - self.font_small.size(hits_text)[0] - 10, 10)

        # Progress bar
        progress = self.distance_traveled / self.FINISH_LINE_DISTANCE
        bar_width = self.SCREEN_WIDTH - 40
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (20, self.SCREEN_HEIGHT - 32, bar_width, 22), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (22, self.SCREEN_HEIGHT - 30, bar_width-4, 18), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_SNAIL_BODY, (22, self.SCREEN_HEIGHT - 30, max(0, (bar_width-4) * progress), 18), border_radius=5)
        
        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, (0, 0))
            msg = "YOU WON!" if self.won else "GAME OVER"
            color = (0, 255, 0) if self.won else (255, 0, 0)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.hits,
            "distance_traveled": self.distance_traveled,
            "speed": self.speed,
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
        # Can't get observation before reset, so we do a light check here
        assert self.observation_space.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert self.observation_space.dtype == np.uint8
        
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    while not done:
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Keyboard input mapping ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = np.array([movement, 0, 0]) # Space/Shift not used
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame rate control ---
        clock.tick(env.FPS)
        
        if done:
            print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before restarting
            obs, info = env.reset()
            done = False

    env.close()