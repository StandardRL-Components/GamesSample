import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:03:51.534713
# Source Brief: brief_02081.md
# Brief Index: 2081
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race a futuristic die down a hazardous track. Tilt the die to steer and land on "
        "specific faces to activate power-ups like speed boosts, shields, and score multipliers."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to tilt the die, steering it left and right to avoid obstacles."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_TRACK = (60, 70, 90)
    COLOR_OBSTACLE = (100, 120, 200)
    COLOR_OBSTACLE_GLOW = (150, 180, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    
    # Power-up Face Colors
    FACE_COLORS = {
        1: (255, 80, 80),   # Speed Boost (RED)
        2: (80, 255, 80),   # Shield (GREEN)
        3: (255, 255, 80),  # Score Multiplier (YELLOW)
        4: (200, 200, 220), # Neutral (WHITE)
        5: (200, 200, 220), # Neutral (WHITE)
        6: (200, 200, 220)  # Neutral (WHITE)
    }
    FACE_GLOW_COLORS = {
        1: (255, 150, 150),
        2: (150, 255, 150),
        3: (255, 255, 150),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player (Dice)
        self.dice_pos = None
        self.dice_rot = None
        self.dice_angular_vel = None
        self.dice_forward_vel = None
        self.dice_size = 30

        # Track & Obstacles
        self.track_segments = None
        self.track_center_x = self.SCREEN_WIDTH / 2
        self.track_width = 250
        self.vanish_point_y = self.SCREEN_HEIGHT * 0.4

        # Power-ups
        self.shield_active = False
        self.speed_boost_timer = 0
        self.score_mult_timer = 0
        self.combo_multiplier = 1
        self.last_powerup_face = -1

        # Particles & Effects
        self.particles = deque()

        # Reward tracking
        self.last_score_milestone = 0

        # 3D Cube Definition
        s = self.dice_size / 2
        self.cube_vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ])
        self.cube_faces = [
            (0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6),
            (4, 0, 3, 7), (3, 2, 6, 7), (4, 5, 1, 0)
        ]
        # Face Normals: +Z, +X, -Z, -X, +Y, -Y
        self.cube_face_normals = np.array([
            [0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]
        ])
        # Map face index to power-up ID (1-6)
        # Top (+Y) is score mult, Front (+Z) is speed, Right (+X) is shield
        self.face_map = { 4: 3, 0: 1, 1: 2, 2: 4, 3: 5, 5: 6 } 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.dice_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.dice_size*1.5, 50.0])
        self.dice_rot = np.array([0.0, 0.0, 0.0]) # Pitch, Yaw, Roll
        self.dice_angular_vel = np.array([0.0, 0.0, 0.0])
        self.dice_forward_vel = 4.0

        self.shield_active = False
        self.speed_boost_timer = 0
        self.score_mult_timer = 0
        self.combo_multiplier = 1.0
        self.last_powerup_face = -1
        
        self.particles.clear()
        self.last_score_milestone = 0

        self._generate_initial_track()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self._update_game_state(movement)
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, movement):
        # --- Physics and Movement ---
        torque_strength = 0.015
        torque = np.array([0.0, 0.0, 0.0])
        if movement == 1: torque[0] -= torque_strength # Tilt forward (pitch)
        if movement == 2: torque[0] += torque_strength # Tilt backward
        if movement == 3: torque[2] -= torque_strength # Tilt left (roll)
        if movement == 4: torque[2] += torque_strength # Tilt right

        self.dice_angular_vel += torque
        self.dice_angular_vel *= 0.95 # Damping
        self.dice_rot += self.dice_angular_vel

        # Determine forward speed
        base_speed = 4.0
        boost_speed = 8.0
        current_speed = base_speed
        if self.speed_boost_timer > 0:
            current_speed = base_speed + (boost_speed * (self.speed_boost_timer / 120.0))
        
        # Update track segments
        for seg in self.track_segments:
            seg['z'] -= current_speed
            for obs in seg['obstacles']:
                obs['z'] -= current_speed
        
        # Sideways movement from rolling
        self.dice_pos[0] -= self.dice_angular_vel[2] * 200
        self.dice_pos[0] = np.clip(self.dice_pos[0], 0, self.SCREEN_WIDTH)
        
        # --- Track Management ---
        if self.track_segments and self.track_segments[0]['z'] < -10:
            self.track_segments.popleft()
        
        if self.track_segments and self.track_segments[-1]['z'] < self.dice_pos[2] + 200:
            self._generate_track_segment()

        # --- Power-up Timers ---
        self.speed_boost_timer = max(0, self.speed_boost_timer - 1)
        self.score_mult_timer = max(0, self.score_mult_timer - 1)
        if self.score_mult_timer > 0:
            self.combo_multiplier = 2.0
        else:
            self.combo_multiplier = 1.0

        # --- Score Update ---
        self.score += current_speed * 0.1 * self.combo_multiplier

        # --- Collision and Power-up Activation ---
        top_face_id = self._get_top_face()
        activated_powerup = False
        if top_face_id != self.last_powerup_face:
            if top_face_id == 1 and self.speed_boost_timer == 0: # Speed
                self.speed_boost_timer = 120 # 4 seconds
                # sfx: speed boost activate
                activated_powerup = True
            elif top_face_id == 2 and not self.shield_active: # Shield
                self.shield_active = True
                # sfx: shield activate
                activated_powerup = True
            elif top_face_id == 3 and self.score_mult_timer == 0: # Score Multiplier
                self.score_mult_timer = 150 # 5 seconds
                # sfx: multiplier activate
                activated_powerup = True
            self.last_powerup_face = top_face_id if activated_powerup else -1

        self._check_collisions()

        # --- Particle Update ---
        self._update_particles(current_speed)


    def _calculate_reward(self):
        reward = 0.0

        # Continuous reward for forward movement
        reward += 0.01

        # Penalty for being off-center
        center_dist = abs(self.dice_pos[0] - self.track_center_x)
        max_dist = self.track_width / 2
        if center_dist > max_dist * 0.5:
             reward -= 0.05 * (center_dist / max_dist)

        # Event-based rewards are handled in their respective functions
        
        # Score milestone reward
        if self.score // 100 > self.last_score_milestone:
            reward += 5.0 # Scaled down from 100 to fit range
            self.last_score_milestone = self.score // 100

        # Terminal rewards
        if self.game_over:
            reward = -10.0 # Collision penalty
        elif self.steps >= self.MAX_STEPS:
            reward = 10.0 # Survival bonus

        return reward

    def _check_termination(self):
        terminated = self.game_over
        if terminated:
            self.game_over = True # Ensure state consistency
        return terminated

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
            "combo": self.combo_multiplier,
            "shield": self.shield_active
        }

    def _render_game(self):
        self._draw_background_grid()
        
        # Sort all renderable objects (obstacles and player) by Z for correct draw order
        render_queue = []
        for seg in self.track_segments:
            for obs in seg['obstacles']:
                render_queue.append({'type': 'obstacle', 'z': obs['z'], 'data': obs})
        render_queue.append({'type': 'player', 'z': self.dice_pos[2], 'data': None})
        render_queue.sort(key=lambda item: item['z'], reverse=True)
        
        self._draw_track()
        
        for item in render_queue:
            if item['type'] == 'obstacle':
                self._draw_obstacle(item['data'])
            elif item['type'] == 'player':
                self._draw_particles()
                if self.shield_active:
                    self._draw_shield()
                self._draw_dice()

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {int(self.score):07d}"
        self._draw_text(score_text, (15, 10), self.font_main, self.COLOR_TEXT, shadow=True)
        
        # Combo display
        if self.combo_multiplier > 1.0:
            combo_text = f"x{self.combo_multiplier:.1f} COMBO!"
            text_surf = self.font_main.render(combo_text, True, self.FACE_COLORS[3])
            pos = (15, 40)
            self._draw_text(combo_text, pos, self.font_main, self.FACE_COLORS[3], shadow=True)

        # Power-up status icons
        icon_size = 20
        y_pos = 15
        if self.speed_boost_timer > 0:
            pygame.draw.rect(self.screen, self.FACE_COLORS[1], (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size))
            pygame.draw.rect(self.screen, self.FACE_GLOW_COLORS[1], (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size), 2)
            y_pos += 25
        if self.shield_active:
            pygame.draw.rect(self.screen, self.FACE_COLORS[2], (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size))
            pygame.draw.rect(self.screen, (200,255,200), (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size), 2)
            y_pos += 25
        if self.score_mult_timer > 0:
            pygame.draw.rect(self.screen, self.FACE_COLORS[3], (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size))
            pygame.draw.rect(self.screen, self.FACE_GLOW_COLORS[3], (self.SCREEN_WIDTH - 35, y_pos, icon_size, icon_size), 2)
    
    # --- Generation and Management ---

    def _generate_initial_track(self):
        self.track_segments = deque()
        for i in range(20):
            is_safe = i < 5 # First 5 segments are safe
            self._generate_track_segment(is_safe)
    
    def _generate_track_segment(self, safe=False):
        last_z = self.track_segments[-1]['z'] if self.track_segments else -50
        new_z = last_z + 50
        
        obstacles = []
        if not safe:
            # Difficulty scaling
            max_density = 0.5
            current_density = min(max_density, 0.05 + (self.score / 500) * 0.05)
            
            if self.np_random.random() < current_density * 3:
                num_obstacles = self.np_random.integers(1, 4)
                possible_x = np.linspace(-self.track_width/2 * 0.9, self.track_width/2 * 0.9, 5)
                chosen_indices = self.np_random.choice(len(possible_x), num_obstacles, replace=False)
                
                for i in chosen_indices:
                    obs_x = self.track_center_x + possible_x[i]
                    obs_z = new_z + self.np_random.uniform(-20, 20)
                    obs_w = self.np_random.uniform(30, 60)
                    obs_h = self.np_random.uniform(40, 80)
                    obstacles.append({'x': obs_x, 'z': obs_z, 'w': obs_w, 'h': obs_h})
        
        segment = {'z': new_z, 'obstacles': obstacles}
        self.track_segments.append(segment)

    def _check_collisions(self):
        dice_rect = pygame.Rect(
            self.dice_pos[0] - self.dice_size / 2,
            self.dice_pos[1] - self.dice_size / 2,
            self.dice_size, self.dice_size
        )

        for seg in self.track_segments:
            for obs in seg['obstacles']:
                # Check only obstacles that are near the player
                if abs(obs['z'] - self.dice_pos[2]) < 10:
                    p_x, p_y, p_scale = self._project(obs['x'], self.SCREEN_HEIGHT, obs['z'])
                    if p_scale < 0.1: continue

                    obs_w_proj = obs['w'] * p_scale
                    obs_h_proj = obs['h'] * p_scale
                    obs_rect = pygame.Rect(
                        p_x - obs_w_proj / 2,
                        p_y - obs_h_proj,
                        obs_w_proj, obs_h_proj
                    )
                    
                    if dice_rect.colliderect(obs_rect):
                        if self.shield_active:
                            self.shield_active = False
                            obs['z'] += 10000 # "Remove" obstacle
                            # sfx: shield break
                        else:
                            self.game_over = True
                            # sfx: crash
                            return

    # --- 3D and Rendering Helpers ---

    def _project(self, x, y, z):
        """Projects a 3D point to 2D screen space."""
        # Simple perspective projection
        # Avoid division by zero
        z_dist = max(0.1, z)
        scale = 100 / z_dist
        
        screen_x = self.track_center_x + (x - self.track_center_x) * scale
        screen_y = self.vanish_point_y + (y - self.vanish_point_y) * scale
        
        return int(screen_x), int(screen_y), scale

    def _get_rotation_matrix(self, pitch, yaw, roll):
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_r, sin_r = math.cos(roll), math.sin(roll)

        Rx = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        Rz = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
        
        return Rz @ Ry @ Rx

    def _get_top_face(self):
        rot_matrix = self._get_rotation_matrix(*self.dice_rot)
        rotated_normals = self.cube_face_normals @ rot_matrix.T
        top_face_idx = np.argmax(rotated_normals[:, 1]) # Max in world Y direction
        return self.face_map[top_face_idx]

    def _draw_dice(self):
        rot_matrix = self._get_rotation_matrix(*self.dice_rot)
        rotated_vertices = self.cube_vertices @ rot_matrix.T
        
        world_vertices = rotated_vertices + np.array([0, 0, self.dice_pos[2]])
        
        # Project vertices to screen
        screen_points = []
        for v in world_vertices:
            sx, sy, _ = self._project(self.dice_pos[0] + v[0], self.dice_pos[1] - v[1], v[2])
            screen_points.append((sx, sy))

        # Painter's algorithm for faces
        face_depths = []
        for i, face in enumerate(self.cube_faces):
            # Average Z of face vertices
            avg_z = sum(world_vertices[v_idx][2] for v_idx in face) / 4
            face_depths.append((avg_z, i))
        
        face_depths.sort(key=lambda x: x[0], reverse=True)
        
        top_face_id = self._get_top_face()

        for _, face_idx in face_depths:
            face_vertices = self.cube_faces[face_idx]
            points = [screen_points[i] for i in face_vertices]
            
            face_id = self.face_map[face_idx]
            color = self.FACE_COLORS[face_id]
            
            # Check back-face culling
            v0, v1, v2 = points[0], points[1], points[2]
            if (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]) > 0:
                # Glow for active power-up faces
                if face_id in self.FACE_GLOW_COLORS:
                    is_active = (face_id == 1 and self.speed_boost_timer > 0) or \
                                (face_id == 2 and self.shield_active) or \
                                (face_id == 3 and self.score_mult_timer > 0)
                    if is_active:
                        self._draw_glowing_polygon(points, self.FACE_GLOW_COLORS[face_id], color)
                    else:
                        pygame.gfxdraw.filled_polygon(self.screen, points, color)
                else:
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                
                pygame.gfxdraw.aapolygon(self.screen, points, (50,50,50))


    def _draw_background_grid(self):
        for i in range(1, 20):
            # Horizontal lines
            z = 5 * i
            y = self.vanish_point_y + (self.SCREEN_HEIGHT - self.vanish_point_y) * (100 / (100 + z))
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, int(y)), (self.SCREEN_WIDTH, int(y)))
        
        # Vertical lines
        for i in range(-15, 16):
            x = self.track_center_x + i * 50
            start_pos = self._project(x, self.SCREEN_HEIGHT, 5)
            end_pos = self._project(x, self.SCREEN_HEIGHT, 200)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos[:2], end_pos[:2])

    def _draw_track(self):
        # The track is defined by its edges
        left_bottom = (self.track_center_x - self.track_width / 2, self.SCREEN_HEIGHT)
        right_bottom = (self.track_center_x + self.track_width / 2, self.SCREEN_HEIGHT)
        left_top = (self.track_center_x - self.track_width/2 * 0.1, self.vanish_point_y)
        right_top = (self.track_center_x + self.track_width/2 * 0.1, self.vanish_point_y)
        
        pygame.gfxdraw.filled_polygon(self.screen, [left_bottom, right_bottom, right_top, left_top], self.COLOR_TRACK)

    def _draw_obstacle(self, obs):
        p_x, p_y, p_scale = self._project(obs['x'], self.SCREEN_HEIGHT, obs['z'])
        if p_scale < 0.05 or p_y > self.SCREEN_HEIGHT + 50: return

        w = obs['w'] * p_scale
        h = obs['h'] * p_scale

        rect = (p_x - w/2, p_y - h, w, h)
        
        # Glow
        glow_rect = (rect[0] - 2, rect[1] - 2, rect[2] + 4, rect[3] + 4)
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=3)
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
        # Highlight
        highlight_rect = (rect[0] + 2, rect[1] + 2, rect[2] - 4, 5)
        pygame.draw.rect(self.screen, (150, 180, 255, 100), highlight_rect, border_radius=2)

    def _draw_shield(self):
        pos = (int(self.dice_pos[0]), int(self.dice_pos[1]))
        radius = int(self.dice_size * 0.8)
        
        # Create a temporary surface for transparency
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (80, 200, 255, 60))
        pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (150, 220, 255, 128))
        self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _update_particles(self, speed):
        # Add new particles for speed trail
        if self.speed_boost_timer > 0:
            for _ in range(2):
                p = {
                    'pos': [self.dice_pos[0] + self.np_random.uniform(-5, 5), self.dice_pos[1] + self.np_random.uniform(-5, 5)],
                    'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)],
                    'life': 20,
                    'color': random.choice([self.FACE_COLORS[1], self.FACE_GLOW_COLORS[1], (255,255,255)])
                }
                self.particles.append(p)
        
        # Update existing particles
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            size = int(p['life'] / 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)

    def _draw_text(self, text, pos, font, color, shadow=False):
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _draw_glowing_polygon(self, points, glow_color, main_color):
        # Simplified glow: draw a larger, transparent polygon behind
        # This is an approximation as pygame doesn't have easy polygon scaling
        centroid = np.mean(points, axis=0)
        glow_points = [centroid + 1.2 * (np.array(p) - centroid) for p in points]
        
        # Use a temporary surface for alpha blending the glow
        min_x = int(min(p[0] for p in glow_points))
        min_y = int(min(p[1] for p in glow_points))
        max_x = int(max(p[0] for p in glow_points))
        max_y = int(max(p[1] for p in glow_points))
        
        if max_x > min_x and max_y > min_y:
            glow_surf = pygame.Surface((max_x - min_x, max_y - min_y), pygame.SRCALPHA)
            local_glow_points = [(p[0] - min_x, p[1] - min_y) for p in glow_points]
            
            try:
                pygame.gfxdraw.filled_polygon(glow_surf, local_glow_points, (*glow_color, 100))
                self.screen.blit(glow_surf, (min_x, min_y))
            except (ValueError, TypeError): # Catch potential errors with malformed polygons
                pass

        pygame.gfxdraw.filled_polygon(self.screen, points, main_color)


    def close(self):
        pygame.quit()

# Example usage for testing
if __name__ == '__main__':
    # The validation code was removed as it's not part of the environment logic
    # and can cause issues in some execution contexts.
    # The main interactive loop is kept for manual testing.
    
    # Set a non-dummy driver for interactive mode
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Override screen for direct rendering
    pygame.display.init()
    pygame.font.init()
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dice Racer")
    
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Manual rendering to the display window
        env.screen.fill(GameEnv.COLOR_BG)
        env._render_game()
        env._render_ui()
        
        if terminated or truncated:
            # Game over screen
            s = pygame.Surface((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            env.screen.blit(s, (0,0))
            env._draw_text("GAME OVER", (GameEnv.SCREEN_WIDTH/2 - 100, GameEnv.SCREEN_HEIGHT/2 - 50), env.font_main, (255,100,100), shadow=True)
            env._draw_text(f"Final Score: {int(info['score'])}", (GameEnv.SCREEN_WIDTH/2 - 120, GameEnv.SCREEN_HEIGHT/2), env.font_main, GameEnv.COLOR_TEXT, shadow=True)
            env._draw_text("Press 'R' to Restart", (GameEnv.SCREEN_WIDTH/2 - 110, GameEnv.SCREEN_HEIGHT/2 + 40), env.font_small, GameEnv.COLOR_TEXT, shadow=True)


        pygame.display.flip()
        env.clock.tick(GameEnv.TARGET_FPS)
        
    env.close()