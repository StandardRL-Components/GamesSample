import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:29:30.810089
# Source Brief: brief_00489.md
# Brief Index: 489
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Reflect a beam of light using rotatable mirrors to hit the target. "
        "Pass through checkpoints to gain more mirrors and increase your score against the clock."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a mirror. Use ↑↓ arrow keys to rotate the selected mirror."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    TIMER_SECONDS = 45

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_MIRROR = (100, 100, 120)
    COLOR_MIRROR_SELECTED = (255, 255, 0)
    COLOR_TARGET = (255, 50, 50)
    COLOR_CHECKPOINT = (50, 255, 50)
    COLOR_BEAM_SLOW = (0, 150, 255)
    COLOR_BEAM_MID = (255, 255, 0)
    COLOR_BEAM_FAST = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)

    # Game Physics
    BEAM_BASE_SPEED = 8
    MIRROR_ROTATION_SPEED = math.radians(5)
    MIRROR_WIDTH, MIRROR_HEIGHT = 80, 8
    PARTICLE_LIFESPAN = 20
    PARTICLE_SPEED = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_speed = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_speed = pygame.font.SysFont(None, 26)

        # --- Game State Initialization ---
        self.level_mirror_count = 10
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mirrors = []
        self.target = {}
        self.checkpoints = []
        self.beam_head = {}
        self.beam_path = []
        self.beam_speed_multiplier = 1.0
        self.timer = 0
        self.selected_mirror_idx = 0
        self.particles = []
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # No need to call this in __init__
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIMER_SECONDS * self.FPS
        self.selected_mirror_idx = 0
        self.particles = []

        self._spawn_entities()
        self._reset_beam()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Handle Player Actions ---
            self._handle_actions(movement)

            # --- Update Game Logic ---
            self.steps += 1
            self.timer -= 1
            
            # Update beam and collect rewards
            step_reward = self._update_beam()
            reward += step_reward

            self._update_particles()

            # --- Check Termination Conditions ---
            if self.game_over: # Set by _update_beam on target hit
                reward += 100
                terminated = True
                self.level_mirror_count += 2
                # Sound: Level Complete
            elif self.timer <= 0:
                reward -= 100
                terminated = True
                self.game_over = True
                # Sound: Time Up
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --------------------------------------------------------------------------
    # Private Helper Methods: Game Logic
    # --------------------------------------------------------------------------

    def _spawn_entities(self):
        self.mirrors = []
        self.checkpoints = []
        
        occupied_rects = []

        # Spawn mirrors
        for _ in range(self.level_mirror_count):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                angle = self.np_random.uniform(0, 2 * math.pi)
                rect = pygame.Rect(0, 0, self.MIRROR_WIDTH + 20, self.MIRROR_HEIGHT + 20)
                rect.center = pos
                if not any(rect.colliderect(r) for r in occupied_rects):
                    self.mirrors.append({'pos': pos, 'angle': angle, 'width': self.MIRROR_WIDTH, 'height': self.MIRROR_HEIGHT})
                    occupied_rects.append(rect)
                    break
        
        # Spawn target
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 100) # Keep away from beam start
            )
            rect = pygame.Rect(0, 0, 40, 40)
            rect.center = pos
            if not any(rect.colliderect(r) for r in occupied_rects):
                self.target = {'pos': pos, 'radius': 15}
                occupied_rects.append(rect)
                break
        
        # Spawn checkpoints
        for _ in range(self.np_random.integers(1, 4)):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                rect = pygame.Rect(0, 0, 30, 30)
                rect.center = pos
                if not any(rect.colliderect(r) for r in occupied_rects):
                    self.checkpoints.append({'pos': pos, 'radius': 10})
                    occupied_rects.append(rect)
                    break

    def _reset_beam(self):
        start_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 10)
        self.beam_head = {
            'pos': start_pos,
            'dir': pygame.Vector2(0, -1).rotate(self.np_random.uniform(-30, 30))
        }
        self.beam_path = [start_pos]
        self.beam_speed_multiplier = 1.0
        # Sound: Beam Reset/Fizzle

    def _handle_actions(self, movement):
        if not self.mirrors: return
        
        if movement == 1: # Rotate CW
            self.mirrors[self.selected_mirror_idx]['angle'] += self.MIRROR_ROTATION_SPEED
        elif movement == 2: # Rotate CCW
            self.mirrors[self.selected_mirror_idx]['angle'] -= self.MIRROR_ROTATION_SPEED
        elif movement == 3: # Select previous
            self.selected_mirror_idx = (self.selected_mirror_idx - 1) % len(self.mirrors)
        elif movement == 4: # Select next
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
        
        if movement in [1, 2]:
             self.mirrors[self.selected_mirror_idx]['angle'] %= (2 * math.pi)

    def _update_beam(self):
        reward = 0
        beam_travel_vec = self.beam_head['dir'] * self.BEAM_BASE_SPEED * self.beam_speed_multiplier
        start_pos = self.beam_head['pos']
        end_pos = start_pos + beam_travel_vec
        
        closest_hit = {'dist': float('inf'), 'obj': None, 'point': None, 'normal': None}

        # --- Check for collisions ---
        # Mirrors
        for i, mirror in enumerate(self.mirrors):
            corners = self._get_rotated_rect_corners(mirror)
            for j in range(4):
                p1 = corners[j]
                p2 = corners[(j + 1) % 4]
                hit_point = self._get_line_segment_intersection(start_pos, end_pos, p1, p2)
                if hit_point:
                    dist = start_pos.distance_to(hit_point)
                    if dist < closest_hit['dist']:
                        normal = (p2 - p1).rotate(90).normalize()
                        closest_hit = {'dist': dist, 'obj': 'mirror', 'point': hit_point, 'normal': normal}

        # Checkpoints
        for i, cp in enumerate(self.checkpoints):
            if self._line_segment_circle_intersection(start_pos, end_pos, cp['pos'], cp['radius']):
                dist = start_pos.distance_to(cp['pos'])
                if dist < closest_hit['dist']:
                    closest_hit = {'dist': dist, 'obj': ('checkpoint', i), 'point': cp['pos']}

        # Target
        if self._line_segment_circle_intersection(start_pos, end_pos, self.target['pos'], self.target['radius']):
            dist = start_pos.distance_to(self.target['pos'])
            if dist < closest_hit['dist']:
                 closest_hit = {'dist': dist, 'obj': 'target', 'point': self.target['pos']}

        # Walls
        wall_hits = [
            (self._get_line_segment_intersection(start_pos, end_pos, pygame.Vector2(0,0), pygame.Vector2(self.WIDTH,0)), pygame.Vector2(0,1)), # Top
            (self._get_line_segment_intersection(start_pos, end_pos, pygame.Vector2(0,self.HEIGHT), pygame.Vector2(self.WIDTH,self.HEIGHT)), pygame.Vector2(0,-1)), # Bottom
            (self._get_line_segment_intersection(start_pos, end_pos, pygame.Vector2(0,0), pygame.Vector2(0,self.HEIGHT)), pygame.Vector2(1,0)), # Left
            (self._get_line_segment_intersection(start_pos, end_pos, pygame.Vector2(self.WIDTH,0), pygame.Vector2(self.WIDTH,self.HEIGHT)), pygame.Vector2(-1,0)), # Right
        ]
        for hit, normal in wall_hits:
            if hit:
                dist = start_pos.distance_to(hit)
                if dist < closest_hit['dist']:
                    closest_hit = {'dist': dist, 'obj': 'wall', 'point': hit, 'normal': normal}
        
        # --- Process Collision ---
        if closest_hit['obj']:
            self.beam_head['pos'] = closest_hit['point']
            
            if closest_hit['obj'] == 'mirror':
                self.beam_head['dir'] = self.beam_head['dir'].reflect(closest_hit['normal'])
                self.beam_path.append(closest_hit['point'])
                self.beam_speed_multiplier = min(3.0, self.beam_speed_multiplier * 1.1)
                reward += 0.1
                self._create_particles(closest_hit['point'], 20, self._get_beam_color())
                # Sound: Beam Reflection
            
            elif isinstance(closest_hit['obj'], tuple) and closest_hit['obj'][0] == 'checkpoint':
                # Checkpoints don't stop the beam, just trigger an event
                self.checkpoints.pop(closest_hit['obj'][1])
                reward += 5
                self._spawn_new_mirrors(3)
                self.beam_head['pos'] = end_pos # Continue moving
                # Sound: Power-up
            
            elif closest_hit['obj'] == 'target':
                self.beam_path.append(closest_hit['point'])
                self.game_over = True
            
            elif closest_hit['obj'] == 'wall':
                reward -= 0.1
                self._reset_beam()
        else:
            self.beam_head['pos'] = end_pos

        # Keep beam path from getting too long
        if len(self.beam_path) > 50:
            self.beam_path.pop(0)

        return reward

    def _spawn_new_mirrors(self, count):
        # Simplified spawning for mid-game
        for _ in range(count):
             pos = pygame.Vector2(
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 50)
            )
             angle = self.np_random.uniform(0, 2 * math.pi)
             self.mirrors.append({'pos': pos, 'angle': angle, 'width': self.MIRROR_WIDTH, 'height': self.MIRROR_HEIGHT})

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(0.5, self.PARTICLE_SPEED)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.PARTICLE_LIFESPAN,
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95 # Damping
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --------------------------------------------------------------------------
    # Private Helper Methods: Rendering
    # --------------------------------------------------------------------------

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw checkpoints
        for cp in self.checkpoints:
            self._draw_glowing_circle(cp['pos'], cp['radius'], self.COLOR_CHECKPOINT)

        # Draw target
        if self.target:
            self._draw_glowing_circle(self.target['pos'], self.target['radius'], self.COLOR_TARGET)
        
        # Draw mirrors
        for i, mirror in enumerate(self.mirrors):
            is_selected = (i == self.selected_mirror_idx)
            self._draw_mirror(mirror, is_selected)
        
        # Draw beam path
        if len(self.beam_path) > 1:
            color = self._get_beam_color()
            self._draw_glowing_line_strip(self.beam_path + [self.beam_head['pos']], color, 3)

        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / self.PARTICLE_LIFESPAN
            color = (p['color'][0], p['color'][1], p['color'][2], int(255 * life_ratio))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius'] * life_ratio), color)

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {self.timer // self.FPS:02d}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Speed Multiplier
        speed_str = f"SPEED: {self.beam_speed_multiplier:.1f}x"
        speed_surf = self.font_speed.render(speed_str, True, self._get_beam_color())
        self.screen.blit(speed_surf, (10, self.HEIGHT - speed_surf.get_height() - 10))

    def _draw_mirror(self, mirror, is_selected):
        corners = self._get_rotated_rect_corners(mirror)
        int_corners = [(int(p.x), int(p.y)) for p in corners]
        pygame.gfxdraw.filled_polygon(self.screen, int_corners, self.COLOR_MIRROR)
        if is_selected:
            pygame.gfxdraw.aapolygon(self.screen, int_corners, self.COLOR_MIRROR_SELECTED)
            pygame.gfxdraw.aapolygon(self.screen, [(c[0]+dx, c[1]+dy) for c in int_corners for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]], self.COLOR_MIRROR_SELECTED)


    def _draw_glowing_circle(self, pos, radius, color):
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos.x), int(pos.y),
                int(radius + i * 2),
                (color[0], color[1], color[2], alpha)
            )
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _draw_glowing_line_strip(self, points, color, max_width):
        if len(points) < 2: return
        
        for i in range(max_width, 0, -1):
            alpha = int(120 * (1 - i / max_width))
            width = i * 2
            if width > 1:
                pygame.draw.lines(self.screen, (*color, alpha), False, points, width)
        pygame.draw.lines(self.screen, color, False, points, 2)

    def _get_beam_color(self):
        if self.beam_speed_multiplier < 1.5:
            t = self.beam_speed_multiplier - 1.0
            return self._lerp_color(self.COLOR_BEAM_SLOW, self.COLOR_BEAM_MID, t / 0.5)
        else:
            t = (self.beam_speed_multiplier - 1.5)
            return self._lerp_color(self.COLOR_BEAM_MID, self.COLOR_BEAM_FAST, t / 1.5)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer // self.FPS}

    # --------------------------------------------------------------------------
    # Private Helper Methods: Math & Geometry
    # --------------------------------------------------------------------------

    def _get_rotated_rect_corners(self, rect_data):
        w, h = rect_data['width'], rect_data['height']
        center = rect_data['pos']
        angle = rect_data['angle']
        
        corners = [
            pygame.Vector2(-w/2, -h/2), pygame.Vector2(w/2, -h/2),
            pygame.Vector2(w/2, h/2), pygame.Vector2(-w/2, h/2)
        ]
        
        rotated_corners = [p.rotate_rad(angle) + center for p in corners]
        return rotated_corners

    def _get_line_segment_intersection(self, p1, p2, p3, p4):
        # line segment p1-p2 and p3-p4
        den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        if den == 0: return None
        
        t_num = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)
        u_num = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x))
        
        t = t_num / den
        u = u_num / den
        
        if 0 < t < 1 and 0 < u < 1:
            return pygame.Vector2(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
        return None

    def _line_segment_circle_intersection(self, p1, p2, circle_center, r):
        # Check if segment intersects circle
        d = p2 - p1
        f = p1 - circle_center
        
        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - r*r
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return False
        else:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
            # Check for case where segment is fully inside circle
            if t1 < 0 and t2 > 1:
                return True
            return False

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

# ==============================================================================
# Main block for human play testing
# ==============================================================================
if __name__ == "__main__":
    # Ensure the display driver is not dummy for human play
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for display
    pygame.display.set_caption("Light Reflection Gym Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    print("\n---" + GameEnv.game_description)
    print("\n---" + GameEnv.user_guide)
    print("\n--- Extra Controls ---")
    print("R: Reset Environment")
    print("Q: Quit")
    print("----------------\n")

    while not terminated:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment Reset. Initial Info: {info}")
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        action = [movement, 0, 0] # space/shift not used
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        if term:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Reset on termination to continue playing
            obs, info = env.reset()
            total_reward = 0
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    pygame.quit()