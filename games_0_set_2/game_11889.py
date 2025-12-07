import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:45:18.165421
# Source Brief: brief_01889.md
# Brief Index: 1889
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls a dynamically extending and
    retracting line from the center of the screen. The goal is to survive
    for as long as possible and complete three levels by avoiding moving circles.

    The game prioritizes visual quality and "game feel" with smooth,
    anti-aliased graphics, a minimalist aesthetic, and responsive controls.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a rotating and extending line from the center of the screen. "
        "Survive as long as possible by avoiding the moving circles."
    )
    user_guide = (
        "Controls: Use ↑/↓ to rotate the line and ←/→ to extend or retract it. "
        "Avoid the red circles."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.LEVEL_DURATION_STEPS = 900 # 30 seconds at 30 FPS
        self.NUM_LEVELS = 3

        # --- Colors ---
        self.COLOR_BG = (26, 26, 46) # Dark blue/purple
        self.COLOR_PLAYER = (0, 255, 153)
        self.COLOR_PLAYER_GLOW = (0, 255, 153, 50)
        self.COLOR_ENEMY = (255, 51, 102)
        self.COLOR_ENEMY_OUTLINE = (255, 150, 170)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_SYNC_BAR = (51, 153, 255)
        self.COLOR_BOUNDARY = (255, 255, 255)

        # --- Gymnasium Spaces ---
        # The observation space is the screen pixels
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        # Action space: [movement, action1, action2]
        # movement: 0=No-op, 1=Rotate CCW, 2=Rotate CW, 3=Extend, 4=Retract
        # action1/2 are unused but required by the spec
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.level = 1
        self.level_timer = 0
        self.line_anchor = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.line_angle = 0.0
        self.line_length = 0.0
        self.consecutive_extends = 0
        self.circles = []
        self.stars = []
        self.last_sync_percentage = 0.0
        self.total_reward = 0.0

        # Initialize state variables and create initial stars
        self._create_stars(100)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.total_reward = 0.0
        self.game_over = False
        self.win = False

        self.level = 1
        self.level_timer = self.LEVEL_DURATION_STEPS

        self.line_angle = self.np_random.uniform(-math.pi, math.pi)
        self.line_length = 50.0
        self.consecutive_extends = 0
        self.last_sync_percentage = 0.0

        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.win:
            # If the game is already over, do nothing and return a terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_circles()
        self._update_level_timer()

        # --- Check for Collisions ---
        collision = self._check_collisions()
        if collision:
            self.game_over = True
            reward = -100.0
        else:
            reward = self._calculate_reward()

        self.score += reward
        self.total_reward += reward
        terminated = self.game_over or self.win
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        # Reset consecutive extends if not extending
        if movement != 3: # 3 is extend
            self.consecutive_extends = 0

        if movement == 1: # Up -> Rotate CCW
            self.line_angle -= math.radians(1.5)
        elif movement == 2: # Down -> Rotate CW
            self.line_angle += math.radians(1.5)
        elif movement == 3: # Left -> Extend
            self.consecutive_extends += 1
            extension_speed = 1.0 + 0.1 * (self.consecutive_extends // 10)
            self.line_length += extension_speed
        elif movement == 4: # Right -> Retract
            retraction_speed = 5.0
            self.line_length -= retraction_speed

        # Clamp line angle and length
        self.line_angle %= (2 * math.pi)
        self.line_length = max(0, self.line_length)

    def _update_circles(self):
        for circle in self.circles:
            circle['pos'] += circle['vel']
            # Boundary bouncing logic
            if circle['pos'][0] <= circle['radius'] or circle['pos'][0] >= self.WIDTH - circle['radius']:
                circle['vel'][0] *= -1
            if circle['pos'][1] <= circle['radius'] or circle['pos'][1] >= self.HEIGHT - circle['radius']:
                circle['vel'][1] *= -1

    def _update_level_timer(self):
        self.level_timer -= 1
        if self.level_timer <= 0:
            self.level += 1
            if self.level > self.NUM_LEVELS:
                self.win = True
            else:
                self.level_timer = self.LEVEL_DURATION_STEPS
                self._setup_level()
                self.score += 10 # Event reward for level complete

    def _calculate_reward(self):
        # Continuous survival reward
        reward = 0.1

        # Synchronization reward
        if self.circles:
            avg_vec = np.mean([c['pos'] - self.line_anchor for c in self.circles], axis=0)
            avg_angle = math.atan2(avg_vec[1], avg_vec[0])

            angle_diff = abs(self.line_angle - avg_angle)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff) # Handle wraparound

            sync_percentage = max(0, 100 * (1 - angle_diff / math.pi))
            
            # Reward for increasing synchronization
            sync_increase = sync_percentage - self.last_sync_percentage
            if sync_increase > 0:
                reward += 0.05 * sync_increase

            self.last_sync_percentage = sync_percentage
        
        if self.win:
            reward += 100
            
        return reward

    def _check_collisions(self):
        p1 = self.line_anchor
        p2 = p1 + self.line_length * np.array([math.cos(self.line_angle), math.sin(self.line_angle)])
        line_vec = p2 - p1
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0: # Zero-length line can't collide
            return False

        for circle in self.circles:
            circle_pos = circle['pos']
            circle_radius = circle['radius']

            # Project circle center onto the line containing the segment
            t = np.dot(circle_pos - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1) # Clamp to the segment

            closest_point = p1 + t * line_vec
            distance_sq = np.sum((circle_pos - closest_point)**2)

            if distance_sq < circle_radius**2:
                return True
        return False

    def _setup_level(self):
        self.circles.clear()
        num_circles = 8 + 2 * self.level # 10, 12, 14
        base_speed = 0.8 + 0.2 * self.level # 1.0, 1.2, 1.4
        
        for _ in range(num_circles):
            # Spawn circles away from the center
            while True:
                pos = self.np_random.uniform([0, 0], [self.WIDTH, self.HEIGHT], size=2)
                if np.linalg.norm(pos - self.line_anchor) > 100:
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(base_speed * 0.8, base_speed * 1.2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            radius = self.np_random.integers(8, 12, endpoint=True)
            
            self.circles.append({'pos': pos, 'vel': vel, 'radius': radius})

    def _create_stars(self, num_stars):
        # This uses python's random, so it won't be seeded by the env.
        # This is fine for non-gameplay cosmetic elements.
        for _ in range(num_stars):
            self.stars.append({
                'pos': [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
                'size': random.uniform(0.5, 1.5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's surfarray has dimensions (width, height, channels).
        # We need to transpose to (height, width, channels) for Gymnasium.
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "level": self.level,
            "win": self.win,
        }

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

    # --- Rendering Sub-routines ---

    def _render_background(self):
        for star in self.stars:
            # Slow parallax effect
            star['pos'][0] = (star['pos'][0] - 0.1 * star['size']) % self.WIDTH
            color_val = int(50 + 50 * star['size'])
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), star['pos'], star['size'])

    def _render_game(self):
        # Render circles
        for circle in self.circles:
            pos = circle['pos'].astype(int)
            radius = int(circle['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY_OUTLINE)

        # Render player line
        p1 = self.line_anchor.astype(int)
        p2 = (self.line_anchor + self.line_length * np.array([math.cos(self.line_angle), math.sin(self.line_angle)]))
        
        # Clamp endpoint to screen boundaries
        p2[0] = np.clip(p2[0], 0, self.WIDTH - 1)
        p2[1] = np.clip(p2[1], 0, self.HEIGHT - 1)
        p2 = p2.astype(int)
        
        if self.line_length > 0:
            # Glow effect
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, p1, p2, width=7)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, p1, p2, width=4)
            # Main line with anti-aliasing
            pygame.gfxdraw.line(self.screen, p1[0], p1[1], p2[0], p2[1], self.COLOR_PLAYER)
        
        # Anchor point
        pygame.gfxdraw.filled_circle(self.screen, p1[0], p1[1], 5, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, p1[0], p1[1], 5, self.COLOR_PLAYER)

    def _render_ui(self):
        # Level Text
        level_text = self.font_medium.render(f"Level: {self.level}/{self.NUM_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Score Text
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(score_text, score_rect)

        # Synchronization Bar
        sync_text = self.font_medium.render("Sync", True, self.COLOR_UI_TEXT)
        sync_rect = sync_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(sync_text, sync_rect)
        
        bar_width = 100
        bar_height = 10
        bar_x = self.WIDTH - 10 - bar_width
        bar_y = sync_rect.bottom + 5
        fill_width = int(bar_width * (self.last_sync_percentage / 100.0))

        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SYNC_BAR, (bar_x, bar_y, fill_width, bar_height))
        
        # Game Over / Win Text
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, rect)
        elif self.win:
            text = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, rect)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Dodge")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Map keyboard inputs to actions
        # 0=No-op, 1=Rotate CCW (Up), 2=Rotate CW (Down), 3=Extend (Left), 4=Retract (Right)
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        clock.tick(env.metadata['render_fps'])
        
    env.close()