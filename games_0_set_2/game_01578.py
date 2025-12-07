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
        "Controls: Arrow keys to move the drawing cursor. Hold Space to draw a track. "
        "Press Shift to snap the cursor back to the sled."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track to guide your sled across the terrain. Reach the checkpoints and the "
        "finish line before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 180

    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (176, 224, 230)  # Powder Blue
    COLOR_SLED = (255, 69, 0)  # Orangered
    COLOR_SLED_OUTLINE = (139, 0, 0)  # Dark Red
    COLOR_TERRAIN = (139, 69, 19)  # Saddle Brown
    COLOR_TRACK = (30, 30, 30)
    COLOR_CURSOR = (0, 191, 255, 150)  # Deep Sky Blue, semi-transparent
    COLOR_CHECKPOINT = (255, 215, 0)  # Gold
    COLOR_FINISH = (60, 179, 113)  # Medium Sea Green
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    # Physics
    GRAVITY = 0.2
    FRICTION = 0.02
    BOUNCE_FACTOR = 0.4
    CURSOR_SPEED = 8
    SLED_RADIUS = 8
    TRACK_WIDTH = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_speed = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables
        self.np_random = None
        # The reset method is called to initialize the state fully.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        
        self._generate_terrain()
        self._setup_checkpoints_and_finish()

        self.sled_pos = np.array([50.0, self.terrain_points[1][1] - 30.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.sled_angle = 0.0
        self.on_track = False
        
        self.drawn_track = []
        self.cursor_pos = self.sled_pos.copy() + np.array([20, 0])
        self.last_cursor_pos = self.cursor_pos.copy()
        
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        previous_x_pos = self.sled_pos[0]
        self._update_sled_physics()
        
        self._update_particles()
        
        self.time_left -= 1
        self.steps += 1
        
        reward += self._calculate_reward(previous_x_pos)
        
        self._check_game_events()
        
        terminated = self.game_over

        if terminated:
            if self.sled_pos[0] >= self.finish_line_x:
                reward += 100 # Finish line bonus
            elif self.time_left <= 0:
                reward -= 10 # Timeout penalty
            else:
                reward -= 50 # Crash penalty

        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        self.last_cursor_pos = self.cursor_pos.copy()
        
        if shift_held:
            # Snap cursor to sled
            self.cursor_pos = self.sled_pos.copy() + np.array([math.cos(self.sled_angle), math.sin(self.sled_angle)]) * 20
        else:
            # Move cursor
            if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
            elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
            elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
            elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED

        # Clamp cursor to screen
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Draw track if space is held and cursor moved
        if space_held and np.linalg.norm(self.cursor_pos - self.last_cursor_pos) > 1:
            # sfx: draw_sound
            self.drawn_track.append((self.last_cursor_pos.copy(), self.cursor_pos.copy()))
            if len(self.drawn_track) > 200: # Limit track segments to prevent memory issues
                self.drawn_track.pop(0)

    def _update_sled_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        self.on_track = False
        on_ground = False  # Local flag for this frame

        # Collision with drawn track
        for p1, p2 in reversed(self.drawn_track):
            d_sq = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
            if d_sq == 0: continue

            # Find closest point on line segment
            t = ((self.sled_pos[0] - p1[0]) * (p2[0] - p1[0]) + (self.sled_pos[1] - p1[1]) * (p2[1] - p1[1])) / d_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * (p2 - p1)
            
            dist_vec = self.sled_pos - closest_point
            dist = np.linalg.norm(dist_vec)

            if dist < self.SLED_RADIUS:
                # Collision detected
                self.on_track = True
                
                # Positional correction
                overlap = self.SLED_RADIUS - dist
                self.sled_pos += (dist_vec / dist) * overlap

                # Velocity response
                normal = dist_vec / dist
                velocity_component_normal = np.dot(self.sled_vel, normal)
                
                if velocity_component_normal < 0:
                    v_normal = normal * velocity_component_normal
                    v_tangent = self.sled_vel - v_normal
                    self.sled_vel = v_tangent * (1 - self.FRICTION) - v_normal * self.BOUNCE_FACTOR
                
                self.sled_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                break # Handle one collision per frame

        # Collision with terrain if not on a drawn track
        if not self.on_track:
            for i in range(len(self.terrain_points) - 1):
                p1 = np.array(self.terrain_points[i])
                p2 = np.array(self.terrain_points[i+1])

                if not (min(p1[0], p2[0]) - self.SLED_RADIUS < self.sled_pos[0] < max(p1[0], p2[0]) + self.SLED_RADIUS):
                    continue
                
                d_sq = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
                if d_sq == 0: continue

                t = ((self.sled_pos[0] - p1[0]) * (p2[0] - p1[0]) + (self.sled_pos[1] - p1[1]) * (p2[1] - p1[1])) / d_sq
                t = np.clip(t, 0, 1)
                closest_point = p1 + t * (p2 - p1)
                
                dist_vec = self.sled_pos - closest_point
                dist = np.linalg.norm(dist_vec)

                if dist < self.SLED_RADIUS:
                    on_ground = True
                    overlap = self.SLED_RADIUS - dist
                    self.sled_pos += (dist_vec / dist) * overlap

                    normal = dist_vec / dist
                    velocity_component_normal = np.dot(self.sled_vel, normal)
                    
                    if velocity_component_normal < 0:
                        v_normal = normal * velocity_component_normal
                        v_tangent = self.sled_vel - v_normal
                        self.sled_vel = v_tangent * (1 - self.FRICTION) - v_normal * self.BOUNCE_FACTOR
                    
                    self.sled_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    break

        # Update position
        self.sled_pos += self.sled_vel

        # Update angle if in air (not on track and not on ground)
        is_airborne = not self.on_track and not on_ground
        if is_airborne and np.linalg.norm(self.sled_vel) > 0.5:
            self.sled_angle = math.atan2(self.sled_vel[1], self.sled_vel[0])

        # Spawn particles
        if np.linalg.norm(self.sled_vel) > 1.0:
            for _ in range(2):
                self._spawn_particle()
    
    def _spawn_particle(self):
        if len(self.particles) < 100:
            particle_vel = -self.sled_vel * self.np_random.uniform(0.1, 0.3) + self.np_random.uniform(-0.5, 0.5, 2)
            particle_life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': self.sled_pos.copy(), 'vel': particle_vel, 'life': particle_life,
                'max_life': particle_life, 'color': (200, 200, 200) if self.on_track else (255, 255, 255)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _calculate_reward(self, previous_x_pos):
        reward = 0
        if self.sled_pos[0] > previous_x_pos: reward += 0.1
        if np.linalg.norm(self.sled_vel) < 0.1: reward -= 0.01

        for cp in self.checkpoints:
            if not cp['reached'] and self.sled_pos[0] >= cp['pos'][0]:
                cp['reached'] = True
                reward += 10
                # sfx: checkpoint_sound
        return reward

    def _check_game_events(self):
        is_off_screen = not (0 < self.sled_pos[0] < self.SCREEN_WIDTH and 0 < self.sled_pos[1] < self.SCREEN_HEIGHT)
        is_finished = self.sled_pos[0] >= self.finish_line_x
        is_timeout = self.time_left <= 0
        
        if is_off_screen or is_finished or is_timeout:
            self.game_over = True

    def _get_observation(self):
        self._render_background()
        self._render_terrain()
        self._render_checkpoints_finish()
        self._render_particles()
        self._render_drawn_track()
        self._render_sled()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "checkpoints_reached": sum(1 for cp in self.checkpoints if cp['reached'])
        }

    def _generate_terrain(self):
        self.terrain_points = []
        y = self.SCREEN_HEIGHT * 0.75
        x = -20
        slope = 0
        while x < self.SCREEN_WIDTH + 20:
            self.terrain_points.append((x, y))
            slope += self.np_random.uniform(-0.1, 0.1)
            slope = np.clip(slope, -0.5, 0.5)
            step_x = self.np_random.uniform(20, 50)
            x += step_x
            y += slope * step_x
            y = np.clip(y, self.SCREEN_HEIGHT * 0.5, self.SCREEN_HEIGHT - 20)
        self.terrain_points.append((self.SCREEN_WIDTH + 20, y))

    def _setup_checkpoints_and_finish(self):
        self.checkpoints = [
            {'pos': (self.SCREEN_WIDTH * 0.33, 0), 'reached': False},
            {'pos': (self.SCREEN_WIDTH * 0.66, 0), 'reached': False}
        ]
        self.finish_line_x = self.SCREEN_WIDTH * 0.95

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_terrain(self):
        terrain_poly = self.terrain_points + [(self.SCREEN_WIDTH + 20, self.SCREEN_HEIGHT), (-20, self.SCREEN_HEIGHT)]
        pygame.gfxdraw.filled_polygon(self.screen, terrain_poly, self.COLOR_TERRAIN)
        top_color = tuple(int(c * 0.7) for c in self.COLOR_TERRAIN)
        pygame.draw.aalines(self.screen, top_color, False, self.terrain_points, 2)

    def _render_checkpoints_finish(self):
        pole_height = self.SCREEN_HEIGHT
        for i, cp in enumerate(self.checkpoints):
            color = self.COLOR_CHECKPOINT if not cp['reached'] else tuple(c//2 for c in self.COLOR_CHECKPOINT)
            x = cp['pos'][0]
            pygame.draw.line(self.screen, (192, 192, 192), (x, 0), (x, pole_height), 1)
            pygame.gfxdraw.filled_polygon(self.screen, [(x, 10), (x+15, 15), (x, 20)], color)
        
        pygame.draw.line(self.screen, (192, 192, 192), (self.finish_line_x, 0), (self.finish_line_x, pole_height), 2)
        pygame.gfxdraw.filled_polygon(self.screen, [(self.finish_line_x, 10), (self.finish_line_x+20, 20), (self.finish_line_x, 30)], self.COLOR_FINISH)

    def _render_drawn_track(self):
        for p1, p2 in self.drawn_track:
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, self.TRACK_WIDTH)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            radius = int(alpha * 3)
            if radius > 0:
                color = (*p['color'], int(alpha * 255))
                try:
                    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (radius, radius), radius)
                    self.screen.blit(surf, (int(p['pos'][0]) - radius, int(p['pos'][1]) - radius))
                except (pygame.error, ValueError):
                    pass # Ignore errors if particle is off-screen

    def _render_sled(self):
        w, h = 16, 8
        center, angle = self.sled_pos, self.sled_angle
        points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(angle) - y * math.sin(angle) + center[0]
            ry = x * math.sin(angle) + y * math.cos(angle) + center[1]
            rotated_points.append((int(rx), int(ry)))
            
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED_OUTLINE)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        radius = 8
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(surf, radius, radius, radius, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(surf, radius, radius, radius, (255,255,255,180))
        pygame.draw.line(surf, (255,255,255,180), (radius-4, radius), (radius+4, radius))
        pygame.draw.line(surf, (255,255,255,180), (radius, radius-4), (radius, radius+4))
        self.screen.blit(surf, (x - radius, y - radius))

    def _render_ui(self):
        def draw_text(text, font, pos, color, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0]+2, pos[1]+2))
            self.screen.blit(text_surf, pos)

        time_str = f"TIME: {int(self.time_left / self.FPS):03d}"
        draw_text(time_str, self.font_ui, (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        score_str = f"SCORE: {int(self.score):,}"
        draw_text(score_str, self.font_ui, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        speed = np.linalg.norm(self.sled_vel) * 5
        speed_str = f"{speed:.1f} km/h"
        draw_text(speed_str, self.font_speed, (10, self.SCREEN_HEIGHT - 30), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        for i, cp in enumerate(self.checkpoints):
            indicator_pos = (self.SCREEN_WIDTH // 2 - 20 + i * 30, 15)
            color = self.COLOR_CHECKPOINT if cp['reached'] else (80, 80, 80)
            pygame.draw.circle(self.screen, color, indicator_pos, 8)
            pygame.draw.circle(self.screen, (255,255,255), indicator_pos, 8, 1)

    def close(self):
        pygame.quit()