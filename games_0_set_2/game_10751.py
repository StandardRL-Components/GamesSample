import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:55.760546
# Source Brief: brief_00751.md
# Brief Index: 751
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
        "Reflect a laser beam off rotating mirrors to hit the target. "
        "Use a color transformer to boost your score."
    )
    user_guide = (
        "Controls: Use arrow keys to aim the laser (←→ for faster aiming). "
        "Press space to fire the beam and shift to change the emitter position."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 1000
    GAME_DURATION_SECONDS = 60

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_EMITTER = (255, 255, 0)
    COLOR_AIM_LINE = (255, 255, 0, 100)
    COLOR_BEAM_DEFAULT = (255, 255, 255)
    COLOR_BEAM_BOOST = (255, 50, 50)
    COLOR_MIRROR_FRAME = (100, 100, 110)
    COLOR_MIRROR_SURFACE = (220, 220, 255)
    COLOR_TARGET = (50, 255, 100)
    COLOR_TRANSFORMER = (200, 50, 255)
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        
        self.emitter_pos = (0, 0)
        self.emitter_pos_idx = 0
        self.beam_angle_deg = 0.0
        
        self.beam_fired_this_step = False
        self.beam_path = []
        self.beam_color = self.COLOR_BEAM_DEFAULT
        self.beam_multiplier = 1
        
        self.mirrors = []
        self.target = None
        self.color_transformer = None
        
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Static Game Elements ---
        self.EMITTER_POSITIONS = [
            (50, 200), (50, 100), (50, 300)
        ]

        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.timer = self.GAME_DURATION_SECONDS
        self.game_over = False
        
        self.emitter_pos_idx = 0
        self.emitter_pos = self.EMITTER_POSITIONS[self.emitter_pos_idx]
        self.beam_angle_deg = 45.0
        
        self.beam_fired_this_step = False
        self.beam_path = []
        self.beam_color = self.COLOR_BEAM_DEFAULT
        self.beam_multiplier = 1
        
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        """Initializes the positions of mirrors, target, etc."""
        self.mirrors = []
        mirror_configs = [
            # pos, width, height, angle
            ((150, 100), 80, 6, 45),
            ((250, 300), 80, 6, -30),
            ((350, 150), 100, 6, 90),
            ((450, 250), 60, 6, 15),
            ((200, 200), 80, 6, 0),
            ((400, 50), 80, 6, -60),
            ((500, 350), 100, 6, 100),
        ]
        for i, (pos, w, h, angle) in enumerate(mirror_configs):
            self.mirrors.append({
                'id': f'mirror_{i}',
                'pos': pygame.Vector2(pos),
                'w': w,
                'h': h,
                'angle_deg': angle,
                'hit_this_frame': False
            })

        self.target = {
            'id': 'target',
            'pos': pygame.Vector2(580, 200),
            'radius': 20
        }
        self.color_transformer = {
            'id': 'transformer',
            'rect': pygame.Rect(280, 220, 40, 40)
        }

    def step(self, action):
        reward = 0
        self.beam_fired_this_step = False
        
        self._handle_input(action)
        
        reward += self._update_game_state()
        
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        terminated = self._check_termination()
        
        if terminated and self.timer <= 0 and not self.game_over:
             # Timeout penalty only if not already won
            reward = -100.0
            self.game_over = True
        
        # Clamp reward
        reward = np.clip(reward, -100.0, 200.0)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1

        # --- Aiming ---
        if movement == 1: self.beam_angle_deg -= 1.0   # "Up" -> CCW
        elif movement == 2: self.beam_angle_deg += 1.0 # "Down" -> CW
        elif movement == 3: self.beam_angle_deg -= 5.0 # "Left" -> Fast CCW
        elif movement == 4: self.beam_angle_deg += 5.0 # "Right" -> Fast CW
        self.beam_angle_deg %= 360

        # --- Fire Beam (rising edge detection) ---
        if space_held and not self.prev_space_held:
            self.beam_fired_this_step = True
            # sfx: player_fire_beam.wav

        # --- Change Emitter Position (rising edge detection) ---
        if shift_held and not self.prev_shift_held:
            self.emitter_pos_idx = (self.emitter_pos_idx + 1) % len(self.EMITTER_POSITIONS)
            self.emitter_pos = self.EMITTER_POSITIONS[self.emitter_pos_idx]
            # sfx: emitter_change_pos.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        """Updates particles and calculates beam path if fired."""
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Reset mirror hit flags
        for m in self.mirrors:
            m['hit_this_frame'] = False
        
        if self.beam_fired_this_step:
            return self._calculate_beam_path()
        return 0.0

    def _calculate_beam_path(self):
        """The core raycasting and reflection logic."""
        self.beam_path = [pygame.Vector2(self.emitter_pos)]
        current_pos = pygame.Vector2(self.emitter_pos)
        current_angle_rad = math.radians(self.beam_angle_deg)
        
        step_reward = 0
        max_reflections = 20

        for _ in range(max_reflections):
            ray_dir = pygame.Vector2(math.cos(current_angle_rad), math.sin(current_angle_rad))
            
            # Find closest intersection
            closest_hit = None
            min_dist_sq = float('inf')

            # Check intersections with all objects
            hittable_objects = self.mirrors + [self.target, self.color_transformer]
            for obj in hittable_objects:
                hit_info = self._get_intersection(current_pos, ray_dir, obj)
                if hit_info:
                    dist_sq = (hit_info['point'] - current_pos).length_squared()
                    if dist_sq > 1e-6 and dist_sq < min_dist_sq: # Ensure we move forward
                        min_dist_sq = dist_sq
                        closest_hit = hit_info
            
            if not closest_hit:
                break # No intersections found, beam goes off-screen

            # Process the closest hit
            hit_obj = closest_hit['obj']
            self.beam_path.append(closest_hit['point'])
            
            self._create_particles(closest_hit['point'], 10, closest_hit.get('color', self.beam_color))

            if hit_obj['id'] == 'target':
                step_reward += 100 * self.beam_multiplier
                self.score += 100 * self.beam_multiplier
                self.game_over = True
                # sfx: target_hit_success.wav
                break

            elif hit_obj['id'].startswith('mirror'):
                step_reward += 0.1
                self.score += 1
                hit_obj['angle_deg'] = (hit_obj['angle_deg'] + 15) % 360
                hit_obj['hit_this_frame'] = True
                
                # Reflection
                normal = closest_hit['normal'].normalize()
                incident = ray_dir
                new_dir = incident - 2 * incident.dot(normal) * normal
                current_angle_rad = math.atan2(new_dir.y, new_dir.x)
                current_pos = closest_hit['point']
                # sfx: mirror_reflect.wav

            elif hit_obj['id'] == 'transformer':
                if self.beam_multiplier == 1:
                    step_reward += 1.0
                    self.score += 10
                    self.beam_color = self.COLOR_BEAM_BOOST
                    self.beam_multiplier = 2
                    # sfx: color_transform.wav
                # Beam passes through, continue from hit point with same angle
                current_pos = closest_hit['point']
        
        return step_reward
        
    def _get_intersection(self, ray_origin, ray_dir, obj):
        """Calculates intersection point of a ray with a game object."""
        obj_id = obj['id']
        
        if obj_id.startswith('mirror'):
            # Model mirror as a line segment
            center = obj['pos']
            w, h = obj['w'], obj['h']
            angle_rad = math.radians(obj['angle_deg'])
            
            p1 = center + pygame.Vector2(-w/2, 0).rotate_rad(angle_rad)
            p2 = center + pygame.Vector2(w/2, 0).rotate_rad(angle_rad)

            v_seg = p2 - p1
            v_ray = ray_dir
            
            p_start = ray_origin
            q_start = p1

            cross_product = v_ray.x * v_seg.y - v_ray.y * v_seg.x
            if abs(cross_product) < 1e-6: return None # Parallel lines

            t = ((q_start.x - p_start.x) * v_seg.y - (q_start.y - p_start.y) * v_seg.x) / cross_product
            u = ((q_start.x - p_start.x) * v_ray.y - (q_start.y - p_start.y) * v_ray.x) / cross_product

            if t > 1e-6 and 0 <= u <= 1:
                point = p_start + t * v_ray
                normal = pygame.Vector2(v_seg.y, -v_seg.x)
                return {'point': point, 'normal': normal, 'obj': obj}
            return None

        elif obj_id == 'target':
            # Ray-circle intersection
            oc = ray_origin - obj['pos']
            a = ray_dir.dot(ray_dir)
            b = 2.0 * oc.dot(ray_dir)
            c = oc.dot(oc) - obj['radius']**2
            discriminant = b*b - 4*a*c
            if discriminant < 0:
                return None
            else:
                t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                if t > 1e-6:
                    point = ray_origin + t * ray_dir
                    return {'point': point, 'obj': obj, 'color': self.COLOR_TARGET}
                return None

        elif obj_id == 'transformer':
            # Ray-AABB intersection
            rect = obj['rect']
            t_min = (rect.left - ray_origin.x) / ray_dir.x if ray_dir.x != 0 else float('-inf')
            t_max = (rect.right - ray_origin.x) / ray_dir.x if ray_dir.x != 0 else float('inf')
            if t_min > t_max: t_min, t_max = t_max, t_min

            ty_min = (rect.top - ray_origin.y) / ray_dir.y if ray_dir.y != 0 else float('-inf')
            ty_max = (rect.bottom - ray_origin.y) / ray_dir.y if ray_dir.y != 0 else float('inf')
            if ty_min > ty_max: ty_min, ty_max = ty_max, ty_min
            
            if (t_min > ty_max) or (ty_min > t_max): return None
            
            t_enter = max(t_min, ty_min)
            if t_enter > 1e-6:
                point = ray_origin + t_enter * ray_dir
                return {'point': point, 'obj': obj, 'color': self.COLOR_TRANSFORMER}
            return None
        
        return None

    def _check_termination(self):
        return self.game_over or self.timer <= 0 or self.steps >= self.MAX_EPISODE_STEPS

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _render_all(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # --- Game Objects ---
        # Color Transformer
        pygame.draw.rect(self.screen, self.COLOR_TRANSFORMER, self.color_transformer['rect'], 0)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_TRANSFORMER), self.color_transformer['rect'], 2)
        
        # Target
        pygame.gfxdraw.filled_circle(self.screen, int(self.target['pos'].x), int(self.target['pos'].y), self.target['radius'], self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, int(self.target['pos'].x), int(self.target['pos'].y), self.target['radius'], tuple(min(255, c+50) for c in self.COLOR_TARGET))

        # Mirrors
        for m in self.mirrors:
            center, w, h, angle = m['pos'], m['w'], m['h'], m['angle_deg']
            points = [
                pygame.Vector2(-w/2, -h/2).rotate(angle) + center,
                pygame.Vector2(w/2, -h/2).rotate(angle) + center,
                pygame.Vector2(w/2, h/2).rotate(angle) + center,
                pygame.Vector2(-w/2, h/2).rotate(angle) + center,
            ]
            pygame.draw.polygon(self.screen, self.COLOR_MIRROR_FRAME, points)
            if m['hit_this_frame']:
                pygame.draw.polygon(self.screen, (255, 255, 100), points, 2) # Highlight on hit
            else:
                pygame.draw.polygon(self.screen, self.COLOR_MIRROR_SURFACE, points, 2)

        # --- Emitter and Aiming Line ---
        pygame.draw.circle(self.screen, self.COLOR_EMITTER, self.emitter_pos, 8)
        pygame.draw.circle(self.screen, self.COLOR_BG, self.emitter_pos, 5)
        
        if not self.beam_fired_this_step:
            angle_rad = math.radians(self.beam_angle_deg)
            end_pos = self.emitter_pos + pygame.Vector2(self.SCREEN_WIDTH, 0).rotate_rad(angle_rad)
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, self.emitter_pos, end_pos)

        # --- Beam ---
        if len(self.beam_path) > 1:
            # Glow effect
            pygame.draw.lines(self.screen, self.beam_color, False, self.beam_path, width=7)
            pygame.draw.lines(self.screen, (min(255,self.beam_color[0]+50), min(255,self.beam_color[1]+50), min(255,self.beam_color[2]+50)), False, self.beam_path, width=3)
        
        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['size'], p['size']))

        # --- UI ---
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        timer_text = self.font_small.render(f"TIME: {max(0, self.timer):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "TARGET HIT!" if self.timer > 0 else "TIME'S UP!"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=self.screen.get_rect().center))

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(10, 20)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'size': random.randint(2, 4),
                'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # To control with keyboard:
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # Space, Shift
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Manual play loop
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Light Bender Gym Environment")
    
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()