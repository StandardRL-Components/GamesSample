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



# Helper class for track segments
class TrackSegment:
    def __init__(self, start_pos, end_pos, segment_type, is_player_drawn=False):
        self.start_pos = pygame.Vector2(start_pos)
        self.end_pos = pygame.Vector2(end_pos)
        self.type = segment_type  # 0: straight, 1: curve up, 2: curve down
        self.width = self.end_pos.x - self.start_pos.x
        self.height_diff = self.end_pos.y - self.start_pos.y
        self.is_player_drawn = is_player_drawn
        self.reward_claimed = False

        # Define curve magnitude based on segment length for consistent feel
        self.curve_magnitude = self.width / 4.0

    def get_y(self, x):
        if self.width <= 0:
            return self.start_pos.y
        
        progress = (x - self.start_pos.x) / self.width
        
        # Linear interpolation for the base height
        base_y = self.start_pos.y + progress * self.height_diff
        
        if self.type == 0: # Straight
            return base_y
        elif self.type == 1: # Curve Up
            return base_y - self.curve_magnitude * math.sin(progress * math.pi)
        elif self.type == 2: # Curve Down
            return base_y + self.curve_magnitude * math.sin(progress * math.pi)
        return base_y

    def get_angle(self, x):
        if self.width <= 0:
            return 0
            
        progress = (x - self.start_pos.x) / self.width
        
        # Derivative of the y-function
        base_slope = self.height_diff / self.width
        
        if self.type == 0: # Straight
            return math.atan(base_slope)
        elif self.type == 1: # Curve Up
            curve_slope = -self.curve_magnitude * (math.pi / self.width) * math.cos(progress * math.pi)
            return math.atan(base_slope + curve_slope)
        elif self.type == 2: # Curve Down
            curve_slope = self.curve_magnitude * (math.pi / self.width) * math.cos(progress * math.pi)
            return math.atan(base_slope + curve_slope)
        return math.atan(base_slope)

    def draw(self, surface, camera_x, color):
        # Draw the segment using a series of short lines for smoothness
        points = []
        start_x_on_screen = int(self.start_pos.x - camera_x)
        end_x_on_screen = int(self.end_pos.x - camera_x)

        # Only draw if visible
        if end_x_on_screen < 0 or start_x_on_screen > surface.get_width():
            return

        for screen_x in range(start_x_on_screen, end_x_on_screen + 1):
            world_x = screen_x + camera_x
            if self.start_pos.x <= world_x <= self.end_pos.x:
                points.append((screen_x, int(self.get_y(world_x))))
        
        if len(points) > 1:
            pygame.draw.aalines(surface, color, False, points)
            # Draw a thicker line underneath for body
            pygame.draw.lines(surface, color, False, [(p[0], p[1]+1) for p in points], 2)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to move drawing cursor. Space to draw track. Shift to cycle track type."
    )

    game_description = (
        "Draw a path for the rider to reach the finish line. Use different track types to navigate the world."
    )

    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRAVITY = 0.3
    FRICTION = 0.998
    RIDER_RADIUS = 10
    HARD_LANDING_VEL_THRESHOLD = 7.0
    MAX_STEPS = 1800 # 60 seconds at 30fps
    FINISH_LINE_X = 5000
    SEGMENT_WIDTH = 100
    DRAW_CURSOR_OFFSET_X = 150
    
    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 58)
    COLOR_RIDER = (0, 200, 255)
    COLOR_RIDER_INNER = (200, 255, 255)
    COLOR_TRACK = (255, 255, 255)
    COLOR_START = (0, 255, 128)
    COLOR_FINISH = (255, 50, 50)
    COLOR_CURSOR = (255, 255, 0, 150) # Yellow, semi-transparent
    COLOR_PARTICLE = (255, 200, 100)
    COLOR_UI_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.track_types = ["STRAIGHT", "CURVE UP", "CURVE DOWN"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.rider_pos = pygame.Vector2(100, 150)
        self.rider_vel = pygame.Vector2(3, 0)
        self.on_track = True
        self.idle_counter = 0

        self.track = []
        # Initial flat track for a safe start
        start_y = 200
        p1 = pygame.Vector2(-20, start_y)
        for i in range(5):
            p2 = pygame.Vector2(p1.x + self.SEGMENT_WIDTH, start_y)
            self.track.append(TrackSegment(p1, p2, 0))
            p1 = p2
        
        self.camera_x = 0
        self.particles = []

        self.cursor_y = self.HEIGHT // 2
        self.current_track_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        step_reward, terminated = self._update_physics()
        reward += step_reward
        
        self._update_world()
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: # Up
            self.cursor_y = max(20, self.cursor_y - 10)
        elif movement == 2: # Down
            self.cursor_y = min(self.HEIGHT - 20, self.cursor_y + 10)

        # Cycle track type (on rising edge)
        if shift_held and not self.last_shift_held:
            self.current_track_type_idx = (self.current_track_type_idx + 1) % len(self.track_types)
            # sfx: UI_cycle.wav

        # Draw track (on rising edge)
        if space_held and not self.last_space_held:
            last_segment_end = self.track[-1].end_pos
            # Only allow drawing if the last segment is somewhat behind the cursor
            if last_segment_end.x < self.rider_pos.x + self.DRAW_CURSOR_OFFSET_X:
                start_pos = last_segment_end
                end_pos = pygame.Vector2(start_pos.x + self.SEGMENT_WIDTH, self.cursor_y)
                new_segment = TrackSegment(start_pos, end_pos, self.current_track_type_idx, is_player_drawn=True)
                self.track.append(new_segment)
                # sfx: draw_track.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_physics(self):
        reward = 0
        terminated = False

        # --- Rider Physics ---
        old_pos = self.rider_pos.copy()
        
        # Apply gravity if not on track
        if not self.on_track:
            self.rider_vel.y += self.GRAVITY
        
        self.rider_vel *= self.FRICTION
        self.rider_pos += self.rider_vel

        # Check for idle termination
        if self.rider_vel.length() < 0.5:
            self.idle_counter += 1
        else:
            self.idle_counter = 0
        if self.idle_counter > 30: # 1 second of being stuck
             terminated = True
             reward = -10

        # --- Track Collision ---
        current_segment = None
        for segment in self.track:
            if segment.start_pos.x <= self.rider_pos.x < segment.end_pos.x:
                current_segment = segment
                break

        was_on_track = self.on_track
        self.on_track = False
        if current_segment:
            track_y = current_segment.get_y(self.rider_pos.x)
            
            if self.rider_pos.y >= track_y - self.RIDER_RADIUS:
                self.on_track = True
                
                # Check for hard landing
                if not was_on_track and self.rider_vel.y > self.HARD_LANDING_VEL_THRESHOLD:
                    reward -= 1
                    self._create_particles(self.rider_pos, 10, self.rider_vel.y / 2)
                    # sfx: hard_landing_sparks.wav

                # Give reward for traversing a player-drawn curve
                if current_segment.is_player_drawn and current_segment.type != 0 and not current_segment.reward_claimed:
                    reward += 5
                    current_segment.reward_claimed = True
                    # sfx: reward_chime.wav
                
                track_angle = current_segment.get_angle(self.rider_pos.x)
                
                # Correct position
                self.rider_pos.y = track_y - self.RIDER_RADIUS

                # Re-orient velocity along track surface
                speed = self.rider_vel.length()
                self.rider_vel.x = math.cos(track_angle) * speed
                self.rider_vel.y = math.sin(track_angle) * speed
                
                # Apply gravitational acceleration along the slope
                self.rider_vel.x += self.GRAVITY * math.sin(track_angle) * math.cos(track_angle)
                self.rider_vel.y += self.GRAVITY * math.sin(track_angle) * math.sin(track_angle)
        
        # Base survival reward
        if not terminated:
            reward += 0.01

        # --- Termination Checks ---
        if self.rider_pos.y > self.HEIGHT + 50 or self.rider_pos.x < self.camera_x - 50:
            terminated = True
            reward = -10
            # sfx: fall_and_crash.wav
        
        if self.rider_pos.x >= self.FINISH_LINE_X:
            terminated = True
            reward = 100
            # sfx: victory_fanfare.wav

        return reward, terminated

    def _update_world(self):
        # Update camera
        self.camera_x = self.rider_pos.x - self.WIDTH / 4

        # Update particles
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # pos.x
            p[1] += p[3] # pos.y
            p[3] += self.GRAVITY / 2 # particle gravity
            p[4] -= 1 # lifetime

        # Prune old track segments
        self.track = [seg for seg in self.track if seg.end_pos.x > self.camera_x - 100]

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
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
        }

    def _render_game(self):
        self._render_background()
        self._render_track()
        self._render_finish_line()
        self._render_particles()
        self._render_rider()
        self._render_cursor()

    def _render_background(self):
        grid_size = 50
        start_x = int(-self.camera_x % grid_size)
        for x in range(start_x, self.WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_track(self):
        for segment in self.track:
            segment.draw(self.screen, self.camera_x, self.COLOR_TRACK)

    def _render_finish_line(self):
        finish_x = int(self.FINISH_LINE_X - self.camera_x)
        if 0 < finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.HEIGHT), 5)
            # Checkered pattern
            for y in range(0, self.HEIGHT, 20):
                pygame.draw.rect(self.screen, self.COLOR_BG, (finish_x - 5, y, 5, 10))
                pygame.draw.rect(self.screen, self.COLOR_BG, (finish_x, y+10, 5, 10))

    def _render_rider(self):
        rider_screen_pos = (int(self.rider_pos.x - self.camera_x), int(self.rider_pos.y))
        
        # Glow effect
        glow_radius = int(self.RIDER_RADIUS * 1.5)
        glow_center = rider_screen_pos
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_RIDER, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Rider body
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        
        # Inner circle for orientation
        if self.rider_vel.length() > 0.1:
            angle = self.rider_vel.angle_to(pygame.Vector2(1,0)) * (math.pi/180.0)
            inner_offset_x = math.cos(angle) * self.RIDER_RADIUS * 0.5
            inner_offset_y = -math.sin(angle) * self.RIDER_RADIUS * 0.5
            pygame.gfxdraw.aacircle(self.screen, int(rider_screen_pos[0] + inner_offset_x), int(rider_screen_pos[1] + inner_offset_y), 3, self.COLOR_RIDER_INNER)
            pygame.gfxdraw.filled_circle(self.screen, int(rider_screen_pos[0] + inner_offset_x), int(rider_screen_pos[1] + inner_offset_y), 3, self.COLOR_RIDER_INNER)

    def _render_cursor(self):
        cursor_screen_x = int(self.rider_pos.x - self.camera_x + self.DRAW_CURSOR_OFFSET_X)
        
        # Draw vertical line
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_screen_x, 0), (cursor_screen_x, self.HEIGHT))
        
        # Draw cursor target
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cursor_screen_x, int(self.cursor_y)), 8, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_screen_x - 12, int(self.cursor_y)), (cursor_screen_x + 12, int(self.cursor_y)), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_screen_x, int(self.cursor_y) - 12), (cursor_screen_x, int(self.cursor_y) + 12), 2)
        
        # Draw preview of the track segment
        if self.track:
            last_segment_end = self.track[-1].end_pos
            if last_segment_end.x < cursor_screen_x + self.camera_x:
                start_pos = last_segment_end
                end_pos = pygame.Vector2(start_pos.x + self.SEGMENT_WIDTH, self.cursor_y)
                preview_segment = TrackSegment(start_pos, end_pos, self.current_track_type_idx)
                
                # Create a temporary surface for transparency
                preview_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                preview_segment.draw(preview_surf, self.camera_x, self.COLOR_CURSOR)
                self.screen.blit(preview_surf, (0, 0))

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p[4] / 5.0))
            screen_pos = (int(p[0] - self.camera_x), int(p[1]))
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (*screen_pos, size, size))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / 30
        time_text = self.font_small.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 40))

        track_type_text = self.font_small.render(f"TRACK: {self.track_types[self.current_track_type_idx]}", True, self.COLOR_UI_TEXT)
        self.screen.blit(track_type_text, (self.WIDTH - track_type_text.get_width() - 10, 10))

        if self.game_over:
            if self.rider_pos.x >= self.FINISH_LINE_X:
                msg = "FINISH!"
            else:
                msg = "CRASHED"
            
            end_text = self.font_large.render(msg, True, self.COLOR_FINISH)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, count, initial_speed):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi) # Upwards
            speed = self.np_random.uniform(1, initial_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pos.x, pos.y, vel.x, vel.y, lifetime])

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    # This part is for human interaction and visualization.
    # It will not be used by the RL agent but is useful for testing.
    import os
    # Set the appropriate video driver for your OS.
    # 'x11', 'dummydriver', 'directfb', 'wayland', 'kmsdrm' for linux
    # 'windows' for windows, 'quartz' for macos
    # If you run into issues, 'dummy' should always work for headless mode.
    # For display, you might need to install additional libraries (e.g., on a server).
    try:
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Track Rider")
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      Track Rider Controls")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered screen, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_reward = 0
                obs, info = env.reset()

    env.close()