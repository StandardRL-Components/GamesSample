import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls mirrors to guide a color-changing
    light beam to hit targets.

    **Visual Style:**
    - Clean, geometric, minimalist with a dark background.
    - Interactive elements (beam, hit targets) are vibrant and glowing.
    - UI is clear and positioned for readability.

    **Gameplay:**
    - The agent selects one of 12 mirrors using up/down actions.
    - The agent rotates the selected mirror using left/right actions.
    - The agent fires a beam of light with the space bar.
    - The beam starts as RED. Each time it reflects off a mirror, its color cycles
      (RED -> BLUE -> GREEN -> RED...).
    - Hitting a target with the beam awards points based on the beam's color:
      - RED: 1 point
      - BLUE: 2 points
      - GREEN: 3 points
    - The goal is to reach 1000 points within 1000 steps.

    **Action Space `MultiDiscrete([5, 2, 2])`:**
    - `action[0]` (Movement):
        - 0: No-op
        - 1: Select previous mirror
        - 2: Select next mirror
        - 3: Rotate selected mirror Counter-Clockwise
        - 4: Rotate selected mirror Clockwise
    - `action[1]` (Space Button):
        - 0: Released
        - 1: Held (fires the beam on the frame it's pressed)
    - `action[2]` (Shift Button):
        - 0/1: No effect (as per brief)
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    game_description = (
        "Guide a color-changing laser beam by rotating mirrors to hit all the targets. "
        "Score points by hitting targets with the correct color."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to select a mirror and ←→ to rotate it. "
        "Press space to fire the laser beam."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    COLOR_BG = (15, 15, 25)
    COLOR_UI_BG = (10, 10, 20)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_MIRROR = (100, 100, 110)
    COLOR_MIRROR_SELECTED = (255, 255, 0)
    COLOR_TARGET_INACTIVE = (60, 60, 70)
    BEAM_COLORS = [(255, 50, 50), (50, 150, 255), (50, 255, 100)]
    
    MAX_STEPS = 1000
    WIN_SCORE = 1000

    NUM_MIRRORS = 12
    NUM_TARGETS = 5
    MIRROR_SIZE = (60, 6)
    TARGET_RADIUS = 15
    BEAM_MAX_BOUNCES = 20

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        self.render_mode = render_mode

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_mirror_idx = 0
        self.last_space_held = False
        
        self.mirrors = []
        self.targets = []
        self.particles = []
        
        self.beam_emitter_pos = pygame.math.Vector2(30, self.GAME_HEIGHT / 2 + self.UI_HEIGHT)
        self.beam_active = False
        self.beam_segments = []
        self.targets_hit_this_shot = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_mirror_idx = 0
        self.last_space_held = False
        
        self.beam_active = False
        self.beam_segments = []
        self.particles.clear()

        self._place_game_elements()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.beam_active = False # Beam is an instantaneous event
        self.beam_segments.clear()

        # Handle actions
        self._handle_actions(movement)

        # Fire beam on space press (rising edge)
        if space_held and not self.last_space_held:
            # SFX: Fire beam
            self.beam_active = True
            reward = self._fire_beam()
            self.score += reward
        
        self.last_space_held = space_held
        self.steps += 1
        
        # Check for termination conditions
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100 # Goal-oriented reward
            # SFX: Win
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _place_game_elements(self):
        self.mirrors.clear()
        self.targets.clear()
        
        # Use a grid to avoid overlaps
        grid_w, grid_h = 10, 6
        cell_w = (self.SCREEN_WIDTH - 100) / grid_w
        cell_h = self.GAME_HEIGHT / grid_h
        occupied_cells = set()

        def get_random_cell():
            while True:
                x = self.np_random.integers(1, grid_w)
                y = self.np_random.integers(0, grid_h)
                if (x, y) not in occupied_cells:
                    occupied_cells.add((x, y))
                    return pygame.math.Vector2(
                        50 + x * cell_w + self.np_random.uniform(-cell_w/4, cell_w/4),
                        self.UI_HEIGHT + y * cell_h + self.np_random.uniform(-cell_h/4, cell_h/4)
                    )

        for _ in range(self.NUM_MIRRORS):
            self.mirrors.append({
                'pos': get_random_cell(),
                'angle': self.np_random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            })
            
        for _ in range(self.NUM_TARGETS):
            self.targets.append({
                'pos': get_random_cell(),
                'hit_color': None
            })

    def _handle_actions(self, movement):
        if movement == 1: # Up
            self.selected_mirror_idx = (self.selected_mirror_idx - 1) % self.NUM_MIRRORS
        elif movement == 2: # Down
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % self.NUM_MIRRORS
        elif movement == 3: # Left (CCW)
            self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] - 45) % 360
            # SFX: Mirror rotate
        elif movement == 4: # Right (CW)
            self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] + 45) % 360
            # SFX: Mirror rotate

    def _fire_beam(self):
        # Reset targets for this shot
        for target in self.targets:
            target['hit_color'] = None
            
        initial_dir = pygame.math.Vector2(1, 0)
        self.beam_segments, hits_info = self._calculate_beam_path(self.beam_emitter_pos, initial_dir)
        
        shot_reward = 0
        num_hits = 0

        for hit in hits_info:
            target_idx = hit['idx']
            color_idx = hit['color_idx']
            
            self.targets[target_idx]['hit_color'] = self.BEAM_COLORS[color_idx]
            
            # Add particles on hit
            # SFX: Target hit
            self._create_particles(self.targets[target_idx]['pos'], self.BEAM_COLORS[color_idx])
            
            shot_reward += (color_idx + 1) # Base reward
            num_hits += 1

        if num_hits == self.NUM_TARGETS:
            shot_reward += 10 # Bonus for clearing all targets
            # SFX: All targets hit bonus
        
        return shot_reward

    def _calculate_beam_path(self, start_pos, start_dir):
        path_segments = []
        hits_info = []
        
        current_pos = pygame.math.Vector2(start_pos)
        current_dir = pygame.math.Vector2(start_dir).normalize()
        current_color_idx = 0

        for _ in range(self.BEAM_MAX_BOUNCES):
            min_dist = float('inf')
            closest_hit = None

            # Check for wall collisions
            for wall_normal, wall_dist_func in [
                (pygame.math.Vector2(1, 0), lambda p: p.x), # Left
                (pygame.math.Vector2(-1, 0), lambda p: self.SCREEN_WIDTH - p.x), # Right
                (pygame.math.Vector2(0, 1), lambda p: p.y - self.UI_HEIGHT), # Top
                (pygame.math.Vector2(0, -1), lambda p: self.SCREEN_HEIGHT - p.y) # Bottom
            ]:
                if current_dir.dot(wall_normal) < 0:
                    dist = wall_dist_func(current_pos) / -current_dir.dot(wall_normal)
                    if 0.001 < dist < min_dist:
                        min_dist = dist
                        closest_hit = {'type': 'wall', 'point': current_pos + dist * current_dir}

            # Check for mirror collisions
            for i, mirror in enumerate(self.mirrors):
                res = self._get_ray_rotated_rect_intersection(current_pos, current_dir, mirror['pos'], self.MIRROR_SIZE, mirror['angle'])
                if res and 0.001 < res['dist'] < min_dist:
                    min_dist = res['dist']
                    closest_hit = {'type': 'mirror', 'point': res['point'], 'normal': res['normal']}

            # Check for target collisions
            for i, target in enumerate(self.targets):
                res = self._get_ray_circle_intersection(current_pos, current_dir, target['pos'], self.TARGET_RADIUS)
                if res and 0.001 < res['dist'] < min_dist:
                    min_dist = res['dist']
                    closest_hit = {'type': 'target', 'point': res['point'], 'idx': i}

            if closest_hit:
                end_point = closest_hit['point']
                path_segments.append({'start': current_pos, 'end': end_point, 'color_idx': current_color_idx})
                
                if closest_hit['type'] == 'wall':
                    break
                elif closest_hit['type'] == 'target':
                    hits_info.append({'idx': closest_hit['idx'], 'color_idx': current_color_idx})
                    break # Beam is absorbed by target
                elif closest_hit['type'] == 'mirror':
                    # SFX: Reflect
                    current_pos = end_point
                    current_dir = current_dir.reflect(closest_hit['normal'])
                    current_color_idx = (current_color_idx + 1) % len(self.BEAM_COLORS)
            else:
                break # No collision found

        return path_segments, hits_info

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()

        # Draw targets
        for target in self.targets:
            color = target['hit_color'] if target['hit_color'] else self.COLOR_TARGET_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, color)

        # Draw mirrors
        for i, mirror in enumerate(self.mirrors):
            points = self._get_rotated_rect_points(mirror['pos'], self.MIRROR_SIZE, mirror['angle'])
            pygame.draw.polygon(self.screen, self.COLOR_MIRROR, points)
            if i == self.selected_mirror_idx:
                pygame.draw.aalines(self.screen, self.COLOR_MIRROR_SELECTED, True, points, 2)
        
        # Draw beam emitter
        pygame.gfxdraw.filled_circle(self.screen, int(self.beam_emitter_pos.x), int(self.beam_emitter_pos.y), 8, self.BEAM_COLORS[0])
        pygame.gfxdraw.aacircle(self.screen, int(self.beam_emitter_pos.x), int(self.beam_emitter_pos.y), 8, (255,255,255))

        # Draw beam if active
        if self.beam_active:
            for seg in self.beam_segments:
                color = self.BEAM_COLORS[seg['color_idx']]
                # Glow effect
                pygame.draw.line(self.screen, color, seg['start'], seg['end'], 5)
                # Core beam
                pygame.draw.aaline(self.screen, (255, 255, 255), seg['start'], seg['end'], 2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Steps
        steps_text = self.font_main.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 8))

        # Beam color indicator
        color_text = self.font_small.render("BEAM", True, self.COLOR_UI_TEXT)
        self.screen.blit(color_text, (200, 12))
        pygame.gfxdraw.filled_circle(self.screen, 250, self.UI_HEIGHT // 2, 10, self.BEAM_COLORS[0])
        pygame.gfxdraw.aacircle(self.screen, 250, self.UI_HEIGHT // 2, 10, (255,255,255))

    def _create_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.uniform(20, 40),
                'color': color
            })
    
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 40))))
                size = int(max(1, 5 * (p['lifespan'] / 40)))
                # Simple rect for performance, circle is slower
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill(p['color'] + (alpha,))
                self.screen.blit(temp_surf, (int(p['pos'].x), int(p['pos'].y)))


    # --- Geometry Helpers ---
    
    @staticmethod
    def _get_rotated_rect_points(center, size, angle_deg):
        w, h = size
        angle_rad = math.radians(angle_deg)
        
        corners = [
            pygame.math.Vector2(-w/2, -h/2), pygame.math.Vector2(w/2, -h/2),
            pygame.math.Vector2(w/2, h/2), pygame.math.Vector2(-w/2, h/2)
        ]
        
        rotated_corners = [c.rotate(-angle_deg) + center for c in corners]
        return [(p.x, p.y) for p in rotated_corners]

    @staticmethod
    def _get_ray_rotated_rect_intersection(ray_origin, ray_dir, rect_center, rect_size, rect_angle_deg):
        # Transform ray into the rectangle's local coordinate system
        p = ray_origin - rect_center
        p = p.rotate(rect_angle_deg)
        d = ray_dir.rotate(rect_angle_deg)
        
        w2, h2 = rect_size[0] / 2, rect_size[1] / 2
        
        min_dist = float('inf')
        hit_normal = None
        
        # Check intersections with the 4 sides of the axis-aligned rect
        for normal, dist_to_origin in [
            (pygame.math.Vector2(-1, 0), w2), (pygame.math.Vector2(1, 0), w2),
            (pygame.math.Vector2(0, -1), h2), (pygame.math.Vector2(0, 1), h2)
        ]:
            if d.dot(normal) != 0:
                t = (dist_to_origin - p.dot(normal)) / d.dot(normal)
                if t > 0.001:
                    hit_point_local = p + t * d
                    # Check if hit point is on the segment
                    other_axis_idx = 1 if normal.x != 0 else 0
                    limit = h2 if other_axis_idx == 1 else w2
                    if abs(hit_point_local[other_axis_idx]) <= limit:
                        if t < min_dist:
                            min_dist = t
                            hit_normal = normal
                            
        if hit_normal:
            return {
                'dist': min_dist,
                'point': ray_origin + min_dist * ray_dir,
                'normal': hit_normal.rotate(-rect_angle_deg)
            }
        return None

    @staticmethod
    def _get_ray_circle_intersection(ray_origin, ray_dir, circle_center, radius):
        oc = ray_origin - circle_center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - radius*radius
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        else:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2.0 * a)
            t2 = (-b + sqrt_d) / (2.0 * a)
            if t1 > 0.001:
                dist = t1
            elif t2 > 0.001:
                dist = t2
            else:
                return None
            return {'dist': dist, 'point': ray_origin + dist * ray_dir}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to select the "human" render mode
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display for manual playing
    pygame.display.set_caption("Laser Maze Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
        
        if terminated:
            # Draw game over screen
            font = pygame.font.Font(None, 50)
            text = font.render(f"Final Score: {info['score']}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH / 2, GameEnv.SCREEN_HEIGHT / 2))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 30)
            reset_text = font_small.render("Press 'R' to reset", True, (200, 200, 200))
            reset_rect = reset_text.get_rect(center=(GameEnv.SCREEN_WIDTH / 2, GameEnv.SCREEN_HEIGHT / 2 + 40))
            screen.blit(reset_text, reset_rect)

            pygame.display.flip()
            clock.tick(30)
            continue

        keys = pygame.key.get_pressed()
        
        # Map keyboard to action space
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()