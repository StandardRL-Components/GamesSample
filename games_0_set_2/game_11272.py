import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Direct a laser beam by adjusting its angle and rotating mirrors to activate all switches, "
        "while avoiding dangerous rotating lasers from platforms."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to aim your laser. Press space and shift to rotate the nearest "
        "platform clockwise and counter-clockwise."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game constants
        self.MAX_STEPS = int(45 * self.metadata["render_fps"]) # 45 seconds at 30 FPS
        self.BEAM_MAX_BOUNCES = 15
        self.PLATFORM_ROTATION_SPEED = math.radians(1.5) # Degrees per step

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_MIRROR = (150, 150, 160)
        self.COLOR_SWITCH_OFF = (0, 70, 200)
        self.COLOR_SWITCH_ON = (0, 200, 255)
        self.COLOR_PLATFORM = (0, 120, 220)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_UI = (240, 240, 240)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.player_pos = None
        self.player_angle = 0.0
        self.mirrors = []
        self.platforms = []
        self.switches = []
        self.beam_path = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.total_switches = 0
        self.activated_switches_count = 0
        self.base_laser_speed = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        self.player_pos = pygame.Vector2(50, self.HEIGHT // 2)
        self.player_angle = self.np_random.uniform(math.pi * -0.25, math.pi * 0.25)

        self._generate_level()

        self.beam_path = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.activated_switches_count = 0
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.mirrors = []
        self.platforms = []
        self.switches = []

        # Generate Mirrors
        for _ in range(self.np_random.integers(4, 7)):
            pos = pygame.Vector2(
                self.np_random.uniform(150, self.WIDTH - 100),
                self.np_random.uniform(50, self.HEIGHT - 50)
            )
            angle = self.np_random.choice([0, math.pi / 2, math.pi / 4, -math.pi / 4])
            self.mirrors.append({'pos': pos, 'size': (10, 80), 'angle': angle})

        # Generate Switches
        for _ in range(self.np_random.integers(3, 5)):
            self.switches.append({
                'pos': pygame.Vector2(
                    self.np_random.uniform(200, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                ),
                'radius': 12,
                'on': False
            })
        self.total_switches = len(self.switches)

        # Generate Platforms
        for _ in range(self.np_random.integers(2, 4)):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(180, self.WIDTH - 80),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                # Ensure platform isn't generated in a way that its laser can
                # immediately hit the static player, ensuring stability at the start.
                if abs(pos.y - self.player_pos.y) > 40:
                    break
            
            self.platforms.append({
                'pos': pos,
                'size': (100, 15),
                'angle': self.np_random.choice([0, math.pi/2]),
                'target_angle': self.np_random.choice([0, math.pi/2]),
                'laser_on': True
            })
        self.base_laser_speed = math.radians(self.np_random.uniform(1.0, 2.0))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for each step
        self.steps += 1
        self.time_left -= 1

        # --- Handle Actions ---
        angle_change = 0
        if movement == 1: angle_change = math.radians(2.0)  # Up
        elif movement == 2: angle_change = math.radians(-2.0) # Down
        elif movement == 3: angle_change = math.radians(-0.5) # Left
        elif movement == 4: angle_change = math.radians(0.5)  # Right
        self.player_angle += angle_change

        # Find closest platform to player for rotation
        if self.platforms:
            closest_platform = min(self.platforms, key=lambda p: self.player_pos.distance_to(p['pos']))
            
            # Rising edge detection for rotation
            if space_held and not self.prev_space_held:
                closest_platform['target_angle'] += math.pi / 2 # Rotate CW
            if shift_held and not self.prev_shift_held:
                closest_platform['target_angle'] -= math.pi / 2 # Rotate CCW

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        # Interpolate platform rotation
        for p in self.platforms:
            angle_diff = (p['target_angle'] - p['angle'] + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) < self.PLATFORM_ROTATION_SPEED:
                p['angle'] = p['target_angle']
            else:
                p['angle'] += math.copysign(self.PLATFORM_ROTATION_SPEED, angle_diff)
        
        # Update laser speed
        laser_speed_increase = 0.02 / self.metadata['render_fps'] * (self.steps // 500)
        current_laser_speed = self.base_laser_speed + laser_speed_increase

        # --- Physics and Collision ---
        self.beam_path, new_switches_on = self._calculate_beam_path()
        
        if new_switches_on > self.activated_switches_count:
            reward += 10.0 * (new_switches_on - self.activated_switches_count)
            self.score += (new_switches_on - self.activated_switches_count)
            self.activated_switches_count = new_switches_on

        # --- Termination Checks ---
        terminated = False
        win = self.activated_switches_count == self.total_switches and self.total_switches > 0
        
        laser_hit = self._check_laser_collision(current_laser_speed)
        timeout = self.time_left <= 0
        
        if win:
            reward += 100.0
            self.score += 100
            terminated = True
        elif laser_hit or timeout:
            reward -= 100.0
            self.score -= 100
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_beam_path(self):
        path = [self.player_pos]
        ray_origin = pygame.Vector2(self.player_pos)
        ray_dir = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle))
        
        collidables = []
        # Mirrors
        for m in self.mirrors:
            points = self._get_rotated_rect_points(m['pos'], m['size'][0], m['size'][1], m['angle'])
            for i in range(2): # Only long sides of mirrors reflect
                p1 = points[i]
                p2 = points[i+1]
                collidables.append({'type': 'mirror', 'p1': p1, 'p2': p2})
        # Platforms
        for p in self.platforms:
            points = self._get_rotated_rect_points(p['pos'], p['size'][0], p['size'][1], p['angle'])
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                collidables.append({'type': 'platform', 'p1': p1, 'p2': p2})

        for s in self.switches: # Reset switches before recalculating
            s['on'] = False

        for _ in range(self.BEAM_MAX_BOUNCES):
            # Switches - checked separately as they don't block
            for s in self.switches:
                if not s['on']:
                    hit = self._ray_circle_intersection(ray_origin, ray_dir, s['pos'], s['radius'])
                    if hit:
                        s['on'] = True

            # Walls and objects
            closest_dist = float('inf')
            closest_hit = None

            for obj in collidables:
                hit_point = self._ray_segment_intersection(ray_origin, ray_dir, obj['p1'], obj['p2'])
                if hit_point:
                    dist = ray_origin.distance_to(hit_point)
                    if dist < closest_dist and dist > 1e-5:
                        closest_dist = dist
                        closest_hit = {'point': hit_point, 'obj': obj}
            
            # Screen boundaries
            for boundary in [((0,0), (self.WIDTH,0)), ((self.WIDTH,0), (self.WIDTH,self.HEIGHT)), ((self.WIDTH,self.HEIGHT), (0,self.HEIGHT)), ((0,self.HEIGHT), (0,0))]:
                hit_point = self._ray_segment_intersection(ray_origin, ray_dir, pygame.Vector2(boundary[0]), pygame.Vector2(boundary[1]))
                if hit_point:
                    dist = ray_origin.distance_to(hit_point)
                    if dist < closest_dist and dist > 1e-5:
                        closest_dist = dist
                        closest_hit = {'point': hit_point, 'obj': {'type': 'wall'}}

            if closest_hit:
                path.append(closest_hit['point'])
                obj_type = closest_hit['obj']['type']
                if obj_type == 'platform' or obj_type == 'wall':
                    break # Beam stops
                elif obj_type == 'mirror':
                    segment_vec = closest_hit['obj']['p2'] - closest_hit['obj']['p1']
                    normal = pygame.Vector2(-segment_vec.y, segment_vec.x).normalize()
                    
                    if ray_dir.dot(normal) > 0:
                        normal = -normal
                    
                    ray_origin = closest_hit['point']
                    ray_dir = ray_dir.reflect(normal)
                else:
                    break
            else:
                break
        
        newly_activated = sum(1 for s in self.switches if s['on'])
        return path, newly_activated

    def _check_laser_collision(self, speed):
        for p in self.platforms:
            if not p['laser_on']: continue
            
            laser_angle = (p['angle'] + (self.steps * speed)) % (2 * math.pi)
            laser_dir = pygame.Vector2(math.cos(laser_angle), math.sin(laser_angle))
            
            # Laser is a ray from platform center. Check distance from player to this ray.
            p_to_player = self.player_pos - p['pos']
            proj = p_to_player.dot(laser_dir)
            if proj > 0: # Player is in front of the laser origin
                dist_sq = p_to_player.magnitude_squared() - proj * proj
                if dist_sq < (10**2): # Player radius of 10
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw mirrors
        for m in self.mirrors:
            points = self._get_rotated_rect_points(m['pos'], m['size'][0], m['size'][1], m['angle'])
            pygame.draw.polygon(self.screen, self.COLOR_MIRROR, [p for p in points])

        # Draw platforms and lasers
        laser_pulse = 0.6 + 0.4 * math.sin(self.steps * 0.2)
        for p in self.platforms:
            points = self._get_rotated_rect_points(p['pos'], p['size'][0], p['size'][1], p['angle'])
            pygame.draw.polygon(self.screen, self.COLOR_PLATFORM, [p for p in points])
            
            laser_angle = (p['angle'] + (self.steps * (self.base_laser_speed + 0.02 / self.metadata['render_fps'] * (self.steps // 500)))) % (2 * math.pi)
            laser_end = p['pos'] + pygame.Vector2(math.cos(laser_angle), math.sin(laser_angle)) * 1000
            self._draw_glowing_line(self.screen, self.COLOR_LASER, p['pos'], laser_end, 2, 4, laser_pulse)

        # Draw switches
        for s in self.switches:
            color = self.COLOR_SWITCH_ON if s['on'] else self.COLOR_SWITCH_OFF
            self._draw_glowing_circle(self.screen, color, s['pos'], s['radius'], 4)

        # Draw beam path
        if len(self.beam_path) > 1:
            for i in range(len(self.beam_path) - 1):
                self._draw_glowing_line(self.screen, self.COLOR_PLAYER, self.beam_path[i], self.beam_path[i+1], 3, 5)

        # Draw player origin
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.player_pos, 8, 4)

    def _render_ui(self):
        # Score/Switches
        switch_text = f"Switches: {self.activated_switches_count}/{self.total_switches}"
        text_surf = self.font_ui.render(switch_text, True, self.COLOR_UI)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        time_str = f"Time: {max(0, self.time_left / self.metadata['render_fps']):.1f}"
        text_surf = self.font_ui.render(time_str, True, self.COLOR_UI)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            win = self.activated_switches_count == self.total_switches and self.total_switches > 0
            msg = "LEVEL COMPLETE" if win else "GAME OVER"
            color = self.COLOR_SWITCH_ON if win else self.COLOR_LASER
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "switches_on": self.activated_switches_count,
            "total_switches": self.total_switches,
            "time_left": self.time_left
        }

    # --- Helper Methods ---
    def _get_rotated_rect_points(self, center, w, h, angle):
        corners = [
            pygame.Vector2(-w/2, -h/2), pygame.Vector2(w/2, -h/2),
            pygame.Vector2(w/2, h/2), pygame.Vector2(-w/2, h/2)
        ]
        return [center + c.rotate_rad(angle) for c in corners]

    def _ray_segment_intersection(self, ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = pygame.Vector2(-ray_dir.y, ray_dir.x)
        dot = v2.dot(v3)
        if abs(dot) < 1e-6: return None
        t1 = v2.cross(v1) / dot
        t2 = v1.dot(v3) / dot
        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return ray_origin + t1 * ray_dir
        return None

    def _ray_circle_intersection(self, ray_origin, ray_dir, circle_center, radius):
        oc = ray_origin - circle_center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - radius*radius
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return False
        else:
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            return t > 0

    def _draw_glowing_line(self, surface, color, start, end, width, glow_layers, pulse=1.0):
        for i in range(glow_layers, 0, -1):
            alpha = int(80 / glow_layers * (glow_layers - i + 1) * pulse)
            glow_color = (*color, alpha)
            pygame.draw.aaline(surface, glow_color, start, end)
        pygame.draw.line(surface, color, start, end, width)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_layers):
        center_int = (int(center.x), int(center.y))
        for i in range(glow_layers, 0, -1):
            alpha = int(100 / glow_layers * (glow_layers - i + 1))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius + i * 2, glow_color)
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius + i * 2, glow_color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()