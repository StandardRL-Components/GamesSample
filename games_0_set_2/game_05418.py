import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move the cursor. Hold Space to draw a line. Release Space to finish it. Shift to clear all your lines."
    )

    game_description = (
        "Draw lines for a sledder to ride on, navigating a minimalist winter landscape. Reach the finish line as fast as you can to maximize your score."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000  # Termination after this many steps

    # Colors
    COLOR_BG = (235, 245, 255) # Light sky blue/white
    COLOR_HILL_1 = (210, 220, 230)
    COLOR_HILL_2 = (190, 200, 210)
    COLOR_SLEDDER = (231, 76, 60) # Bright Red
    COLOR_SLEDDER_GLOW = (255, 120, 100)
    COLOR_LINE = (40, 40, 40)
    COLOR_DRAWING_LINE = (140, 140, 140)
    COLOR_CURSOR = (46, 204, 113) # Bright Green
    COLOR_FINISH_LINE = (39, 174, 96)
    COLOR_TEXT = (50, 50, 50)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    GRAVITY = 0.2
    FRICTION = 0.995
    CURSOR_SPEED = 8.0
    SLEDDER_RADIUS = 6
    COLLISION_THRESHOLD = 8.0 # How close to a line to be considered "on" it

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 60)
        
        self.finish_line_x = self.WIDTH - 50
        
        # Static background elements
        self.background_hills = self._generate_hills()

        # Initialize state variables
        self.sledder_pos = None
        self.sledder_vel = None
        self.lines = None
        self.initial_line = None
        self.particles = None
        self.cursor_pos = None
        self.is_drawing = None
        self.drawing_start_pos = None
        self.prev_space_held = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_outcome = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sledder_pos = pygame.Vector2(80, 50)
        self.sledder_vel = pygame.Vector2(0, 0)
        
        # Pre-drawn initial line
        self.initial_line = (pygame.Vector2(40, 120), pygame.Vector2(180, 110))
        self.lines = [self.initial_line]
        
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.is_drawing = False
        self.drawing_start_pos = None
        self.prev_space_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._handle_input(action)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        
        terminated, outcome = self._check_termination()
        reward = self._calculate_reward(terminated, outcome)
        self.score += reward

        if terminated:
            self.game_over = True
            self.game_outcome = outcome
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
             self.game_over = True
             self.game_outcome = "TIMEOUT"
             terminated = True # Per Gymnasium API, terminated should be True if truncated is
             
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED # Up
        if movement == 2: self.cursor_pos.y += self.CURSOR_SPEED # Down
        if movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED # Left
        if movement == 4: self.cursor_pos.x += self.CURSOR_SPEED # Right
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # --- Line Drawing ---
        if space_held and not self.prev_space_held:
            # SFX: Pen down
            self.is_drawing = True
            self.drawing_start_pos = pygame.Vector2(self.cursor_pos)
        elif not space_held and self.prev_space_held and self.is_drawing:
            # SFX: Line drawn
            self.is_drawing = False
            end_pos = pygame.Vector2(self.cursor_pos)
            if self.drawing_start_pos and self.drawing_start_pos.distance_to(end_pos) > 5: # Min line length
                self.lines.append((self.drawing_start_pos, end_pos))

        self.prev_space_held = space_held

        # --- Clear Lines ---
        if shift_held:
            # SFX: Erase sound
            self.lines = [self.initial_line]

    def _update_physics(self):
        was_on_ground = hasattr(self, 'on_ground') and self.on_ground

        # Find the highest ground point beneath the sledder
        ground_point = None
        ground_line_vec = None
        min_dist_sq = float('inf')

        for line_start, line_end in self.lines:
            p, d_sq = self._get_closest_point_on_segment(self.sledder_pos, line_start, line_end)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                # Check if sledder is roughly "above" the line segment
                if self.sledder_pos.y <= p.y + self.COLLISION_THRESHOLD:
                    ground_point = p
                    ground_line_vec = line_end - line_start

        self.on_ground = ground_point is not None and min_dist_sq < self.COLLISION_THRESHOLD**2

        if self.on_ground:
            # --- On a line ---
            self.sledder_pos.y = ground_point.y # Snap to the line
            
            if not was_on_ground and self.sledder_vel.length() > 1.0:
                # SFX: Snow crunch on landing
                self._spawn_particles(self.sledder_pos, 20, self.sledder_vel.length() / 2)

            line_angle = math.atan2(ground_line_vec.y, ground_line_vec.x)
            
            # Acceleration along the slope
            accel = self.GRAVITY * math.sin(line_angle)
            self.sledder_vel.x += accel * math.cos(line_angle)
            self.sledder_vel.y += accel * math.sin(line_angle)

            # Apply friction
            self.sledder_vel *= self.FRICTION

        else:
            # --- In the air ---
            # SFX: Air whoosh
            self.sledder_vel.y += self.GRAVITY

        self.sledder_pos += self.sledder_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Air drag
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _spawn_particles(self, pos, count, speed_multiplier):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_multiplier
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed - 1.0)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': random.randint(20, 40),
                'radius': random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.sledder_pos.x >= self.finish_line_x:
            return True, "FINISH"
        if self.sledder_pos.y > self.HEIGHT + 20:
            return True, "FELL"
        if self.steps >= self.MAX_STEPS:
            return True, "TIMEOUT"
        return False, ""

    def _calculate_reward(self, terminated, outcome):
        if terminated:
            if outcome == "FINISH":
                time_bonus = max(0, 50 * ((self.MAX_STEPS / 3) - self.steps) / (self.MAX_STEPS / 3))
                return 5 + time_bonus
            if outcome == "FELL":
                return -100.0
            if outcome == "TIMEOUT":
                return -50.0
        
        # Reward for moving forward
        if self.sledder_vel.x > 0:
            return 0.1
            
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_lines()
        self._render_particles()
        self._render_sledder_and_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for hill in self.background_hills:
            pygame.gfxdraw.filled_polygon(self.screen, hill['points'], hill['color'])
        
        # Start and Finish Lines
        pygame.draw.line(self.screen, self.COLOR_LINE, (40, 0), (40, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (self.finish_line_x, 0), (self.finish_line_x, self.HEIGHT), 4)

    def _render_lines(self):
        for start, end in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, start, end, 3)
        if self.is_drawing and self.drawing_start_pos:
            pygame.draw.aaline(self.screen, self.COLOR_DRAWING_LINE, self.drawing_start_pos, self.cursor_pos, 2)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            # Pygame doesn't handle alpha in filled_circle well without a separate surface, so we skip it for simplicity
            color = self.COLOR_PARTICLE
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            
    def _render_sledder_and_cursor(self):
        # Sledder
        sled_pos = (int(self.sledder_pos.x), int(self.sledder_pos.y))
        # Pygame doesn't handle alpha in filled_circle well, so we draw a solid glow circle
        pygame.gfxdraw.filled_circle(self.screen, sled_pos[0], sled_pos[1], self.SLEDDER_RADIUS + 2, self.COLOR_SLEDDER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, sled_pos[0], sled_pos[1], self.SLEDDER_RADIUS, self.COLOR_SLEDDER)
        pygame.gfxdraw.aacircle(self.screen, sled_pos[0], sled_pos[1], self.SLEDDER_RADIUS, self.COLOR_SLEDDER)
        
        # Cursor
        if not self.game_over:
            cursor_pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0] - 6, cursor_pos_int[1]), (cursor_pos_int[0] + 6, cursor_pos_int[1]), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0], cursor_pos_int[1] - 6), (cursor_pos_int[0], cursor_pos_int[1] + 6), 2)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS / self.FPS) - (self.steps / self.FPS))
        time_text = f"TIME: {time_left:.2f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 35))
        
        # Game Over Text
        if self.game_over:
            outcome_map = {
                "FINISH": "FINISH!",
                "FELL": "GAME OVER",
                "TIMEOUT": "TIME'S UP!"
            }
            color_map = {
                "FINISH": self.COLOR_FINISH_LINE,
                "FELL": self.COLOR_SLEDDER,
                "TIMEOUT": self.COLOR_TEXT
            }
            text = outcome_map.get(self.game_outcome, "GAME OVER")
            color = color_map.get(self.game_outcome, self.COLOR_TEXT)
            
            end_surf = self.font_big.render(text, True, color)
            text_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, text_rect)

    def _generate_hills(self):
        hills = []
        for i in range(3):
            base_y = self.HEIGHT - random.randint(0, 100)
            width = random.randint(200, 400)
            center_x = random.randint(0, self.WIDTH)
            color = self.COLOR_HILL_1 if i % 2 == 0 else self.COLOR_HILL_2
            points = [
                (center_x - width // 2, self.HEIGHT),
                (center_x, base_y),
                (center_x + width // 2, self.HEIGHT)
            ]
            hills.append({'points': points, 'color': color})
        return hills

    def _get_closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_len_sq = ab.length_squared()
        if ab_len_sq == 0:
            return a, p.distance_squared_to(a)
        
        t = ap.dot(ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        
        closest_point = a + ab * t
        return closest_point, p.distance_squared_to(closest_point)

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will create a visible window for interaction.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Line Sledder")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or wait for 'r' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    pygame.quit()