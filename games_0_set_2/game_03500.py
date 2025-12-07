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
    user_guide = (
        "Controls: Use arrow keys to aim the next track segment. Press space to draw it. "
        "Guide the sled to the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based sledding game. Draw the track ahead of your sled to navigate a "
        "procedurally generated mountain and reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game Constants ---
        self.world_width = 100.0  # meters
        self.world_height = 50.0  # meters, for camera scaling
        self.gravity = 9.8
        self.max_steps = 1000
        self.sled_radius = 2.0 # world units
        self.line_draw_length = 8.0 # world units
        self.friction = 0.998
        self.physics_sub_steps = 5

        # --- Colors ---
        self.color_bg_top = (20, 30, 50)
        self.color_bg_bottom = (40, 50, 70)
        self.color_terrain = (60, 70, 90)
        self.color_terrain_line = (100, 110, 130)
        self.color_sled = (255, 255, 255)
        self.color_sled_glow = (200, 200, 255, 50)
        self.color_drawn_line = (255, 50, 50)
        self.color_cursor = (255, 100, 100, 150)
        self.color_start = (50, 200, 50)
        self.color_finish = (50, 50, 255)
        self.color_text = (240, 240, 240)
        self.color_trail = (255, 255, 255)

        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.sled_pos = None
        self.sled_vel = None
        self.sled_trail = None
        self.terrain_points = None
        self.drawn_lines = None
        self.last_line_end = None
        self.cursor_angle = None
        self.last_distance_traveled = None
        self.checkpoints_reached = None
        self.camera_y_offset = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        start_pos = pygame.Vector2(5.0, 15.0)
        self.sled_pos = pygame.Vector2(start_pos)
        self.sled_vel = pygame.Vector2(0, 0)
        self.sled_trail = []

        self._generate_terrain()
        start_line_p1 = pygame.Vector2(start_pos.x - 5, start_pos.y)
        start_line_p2 = pygame.Vector2(start_pos.x + 5, start_pos.y)
        self.drawn_lines = [(start_line_p1, start_line_p2)]
        self.last_line_end = pygame.Vector2(start_line_p2)
        self.cursor_angle = 0

        self.last_distance_traveled = self.sled_pos.x
        self.checkpoints_reached = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0

        # --- Action Handling ---
        angle_change = math.pi / 32
        if movement == 1: self.cursor_angle -= angle_change  # Up -> Steeper
        elif movement == 2: self.cursor_angle += angle_change  # Down -> Flatter
        elif movement == 3: self.cursor_angle -= angle_change * 0.5 # Left
        elif movement == 4: self.cursor_angle += angle_change * 0.5 # Right
        self.cursor_angle = max(-math.pi/2.1, min(math.pi/2.1, self.cursor_angle))

        if space_pressed:
            # sfx: draw_line
            new_line_end = self.last_line_end + pygame.Vector2(
                self.line_draw_length * math.cos(self.cursor_angle),
                self.line_draw_length * math.sin(self.cursor_angle)
            )
            self.drawn_lines.append((self.last_line_end, new_line_end))
            self.last_line_end = new_line_end
            self.cursor_angle = 0
            if len(self.drawn_lines) > 50: self.drawn_lines.pop(0)

        # --- Physics and Game Logic ---
        dt = (1.0 / 30.0) / self.physics_sub_steps
        for _ in range(self.physics_sub_steps):
            if not self.game_over: self._update_physics(dt)

        self.steps += 1
        reward += self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.sled_pos.x >= self.world_width:
                reward += 100  # Victory bonus
            else:
                reward -= 1  # Crash penalty
        
        self.score += reward

        truncated = self.steps >= self.max_steps
        if truncated:
            self.game_over = True
        
        terminated = terminated or truncated

        return (
            self._get_observation(), reward, terminated, truncated, self._get_info()
        )

    def _update_physics(self, dt):
        self.sled_vel.y += self.gravity * dt
        self.sled_trail.append(pygame.Vector2(self.sled_pos))
        if len(self.sled_trail) > 50: self.sled_trail.pop(0)

        for p1, p2 in reversed(self.drawn_lines):
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            if not (min(p1.x, p2.x) <= self.sled_pos.x <= max(p1.x, p2.x)): continue

            t = (self.sled_pos.x - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
            if 0 <= t <= 1:
                line_y = p1.y + t * (p2.y - p1.y)
                if self.sled_pos.y > line_y - self.sled_radius:
                    self.sled_pos.y = line_y - self.sled_radius
                    line_normal = pygame.Vector2(-line_vec.y, line_vec.x).normalize()
                    if line_normal.y > 0: line_normal *= -1
                    
                    vel_dot_normal = self.sled_vel.dot(line_normal)
                    if vel_dot_normal > 0:
                        restitution = 0.3
                        self.sled_vel -= (1 + restitution) * vel_dot_normal * line_normal
                    
                    self.sled_vel *= self.friction
                    break

        self.sled_pos += self.sled_vel * dt

    def _calculate_reward(self):
        reward = 0
        progress = self.sled_pos.x - self.last_distance_traveled
        if progress > 0: reward += progress * 0.1
        self.last_distance_traveled = self.sled_pos.x

        current_checkpoint = int(self.sled_pos.x // 25)
        if current_checkpoint > 0 and current_checkpoint > self.checkpoints_reached:
            # sfx: checkpoint_achieved
            self.checkpoints_reached = current_checkpoint
            reward += 5
        return reward

    def _check_termination(self):
        if self.sled_pos.x >= self.world_width: return True
        
        screen_pos = self._world_to_screen(self.sled_pos)
        if not (0 <= screen_pos.y < self.screen_height + 50): return True
            
        terrain_y = self._get_terrain_height(self.sled_pos.x)
        if self.sled_pos.y > terrain_y - self.sled_radius: return True
            
        return False

    def _world_to_screen(self, world_pos):
        scale_x = self.screen_width / self.world_width
        scale_y = self.screen_height / self.world_height
        screen_x = world_pos.x * scale_x
        screen_y = (world_pos.y - self.camera_y_offset) * scale_y + self.screen_height / 2
        return pygame.Vector2(int(screen_x), int(screen_y))

    def _generate_terrain(self):
        self.terrain_points = []
        num_points = 200
        base_y = self.world_height * 0.8
        waves = [{"amp": self.np_random.uniform(0.5, 3.0),
                  "freq": self.np_random.uniform(0.1, 0.5),
                  "phase": self.np_random.uniform(0, 2 * math.pi)} for _ in range(4)]

        for i in range(num_points + 1):
            x = (i / num_points) * self.world_width
            y = base_y
            difficulty_mod = 1.0 + 0.05 * (x // 25)
            for wave in waves:
                y += wave["amp"] * difficulty_mod * math.sin(wave["freq"] * x + wave["phase"])
            self.terrain_points.append(pygame.Vector2(x, y))

    def _get_terrain_height(self, world_x):
        if not self.terrain_points or not (self.terrain_points[0].x <= world_x <= self.terrain_points[-1].x):
            return self.world_height * 2
        
        for i in range(len(self.terrain_points) - 1):
            p1, p2 = self.terrain_points[i], self.terrain_points[i+1]
            if p1.x <= world_x <= p2.x:
                t = (world_x - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
                return p1.y + t * (p2.y - p1.y)
        return self.world_height * 2

    def _get_observation(self):
        self.camera_y_offset = self.sled_pos.y - self.world_height / 2
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        # Background
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.color_bg_top, self.color_bg_bottom))
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Terrain
        screen_points = [self._world_to_screen(p) for p in self.terrain_points]
        poly_points = screen_points + [(self.screen_width, self.screen_height), (0, self.screen_height)]
        pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.color_terrain)
        if len(screen_points) > 1: pygame.draw.aalines(self.screen, self.color_terrain_line, False, screen_points)

        # Markers
        for i in range(int(self.world_width // 25) + 1):
            x = i * 25
            color = self.color_start if i == 0 else self.color_finish if x >= self.world_width else (200, 200, 200, 100)
            width = 3 if i == 0 or x >= self.world_width else 1
            screen_x = self._world_to_screen(pygame.Vector2(x, 0)).x
            pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, self.screen_height), width)
        
        # Drawn Lines
        for p1, p2 in self.drawn_lines:
            pygame.draw.aaline(self.screen, self.color_drawn_line, self._world_to_screen(p1), self._world_to_screen(p2), 2)

        # Trail
        for i, pos in enumerate(self.sled_trail):
            if i % 2 == 0:
                alpha = int(100 * (i / len(self.sled_trail)))
                screen_pos = self._world_to_screen(pos)
                radius = int(1.5 * (i / len(self.sled_trail)))
                if radius > 0: pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, self.color_trail + (alpha,))

        # Sled
        screen_pos = self._world_to_screen(self.sled_pos)
        radius_px = int(self.sled_radius * (self.screen_width / self.world_width) * 0.5)
        if radius_px > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(radius_px * 2.5), self.color_sled_glow)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius_px, self.color_sled)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), radius_px, (180, 180, 180))

        # Cursor
        if not self.game_over:
            cursor_end = self.last_line_end + pygame.Vector2(self.line_draw_length * math.cos(self.cursor_angle), self.line_draw_length * math.sin(self.cursor_angle))
            start_screen, end_screen = self._world_to_screen(self.last_line_end), self._world_to_screen(cursor_end)
            pygame.draw.aaline(self.screen, self.color_cursor, start_screen, end_screen)
            pygame.gfxdraw.filled_circle(self.screen, int(end_screen.x), int(end_screen.y), 4, self.color_cursor)

        # UI
        dist_surf = self.font_small.render(f"Distance: {self.sled_pos.x:.1f}m", True, self.color_text)
        score_surf = self.font_small.render(f"Score: {int(self.score)}", True, self.color_text)
        self.screen.blit(dist_surf, (10, 10))
        self.screen.blit(score_surf, (10, 30))

        if self.game_over:
            message = "GOAL!" if self.sled_pos.x >= self.world_width else "CRASHED!"
            color = self.color_start if self.sled_pos.x >= self.world_width else self.color_drawn_line
            end_surf = self.font_large.render(message, True, color)
            self.screen.blit(end_surf, end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.sled_pos.x,
            "is_success": self.sled_pos.x >= self.world_width
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play testing and is not part of the Gymnasium environment
    env = GameEnv()
    obs, info = env.reset()
    
    try:
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Sled Rider")
        in_display_mode = True
    except pygame.error:
        print("No display available. Running in headless mode.")
        in_display_mode = False
        
    terminated = False
    truncated = False
    print(GameEnv.user_guide)

    while not terminated and not truncated:
        action = [0, 0, 0] # Default no-op action
        
        if in_display_mode:
            movement, space_pressed = 0, 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        space_pressed = 1
                    elif event.key == pygame.K_r: # Reset on 'r' key
                        obs, info = env.reset()
                        print("Environment reset.")
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if movement != 0 or space_pressed == 1:
                action = [movement, space_pressed, 0]
                obs, reward, terminated, truncated, info = env.step(action)
            else: # If no user input, just render the current state
                 obs = env._get_observation()

            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30)
        else: # Headless mode: step with random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")

    if info:
        print(f"Game Over. Final Score: {info['score']:.0f}, Success: {info['is_success']}")
    
    if in_display_mode:
        pygame.time.wait(2000)

    env.close()