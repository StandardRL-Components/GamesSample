import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:16:27.152408
# Source Brief: brief_01632.md
# Brief Index: 1632
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Gymnasium environment for a light refraction puzzle game.

    The player controls a set of 5 rotatable lenses to guide light beams
    from sources to a target. The goal is to achieve a combined beam
    intensity of 75% or more at the target within a 30-second time limit.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement/Rotation (0=none, 1=coarse rot CW, 2=coarse rot CCW, 3=select prev lens, 4=select next lens)
    - actions[1]: Fine Rotation CW (0=released, 1=held)
    - actions[2]: Fine Rotation CCW (0=released, 1=held)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A light refraction puzzle. Rotate a series of lenses to guide light beams "
        "from their sources to a central target, aiming to maximize the combined intensity."
    )
    user_guide = (
        "Controls: ←→ to select a lens. ↑↓ to rotate it. Hold space for fine clockwise "
        "rotation, or shift for fine counter-clockwise rotation."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 50  # Logical steps per second
        self.MAX_STEPS = 30 * self.FPS  # 30-second limit
        self.WIN_INTENSITY = 0.75  # 75%
        self.LENS_INTENSITY_LOSS = 0.5
        self.NUM_LENSES = 5
        self.NUM_BEAM_SOURCES = 2
        self.MAX_BOUNCES = self.NUM_LENSES + 1

        # --- Colors and Style ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_BEAM = (255, 255, 255)
        self.COLOR_LENS = (100, 200, 255)
        self.COLOR_LENS_SELECTED = (200, 255, 255)
        self.COLOR_TARGET = (255, 255, 255)
        self.COLOR_TARGET_WIN = (100, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)

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
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 24)
        except IOError:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont(None, 48)
            self.font_small = pygame.font.SysFont(None, 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lenses = []
        self.beam_sources = []
        self.target = {}
        self.selected_lens_idx = 0
        self.beam_paths = []
        self.target_intensity = 0.0
        self.last_target_intensity = 0.0
        self.win_event_triggered = False
        self.action_cooldown = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_lens_idx = 0
        self.last_target_intensity = 0.0
        self.win_event_triggered = False
        self.action_cooldown = 0

        # Initialize game entities
        self.target = {'pos': np.array([580, self.HEIGHT / 2]), 'radius': 40}
        
        self.beam_sources = []
        for i in range(self.NUM_BEAM_SOURCES):
            y_pos = self.HEIGHT * (i + 1) / (self.NUM_BEAM_SOURCES + 1)
            self.beam_sources.append({
                'pos': np.array([40, y_pos]),
                'angle': 0.0 # degrees
            })

        self.lenses = []
        for i in range(self.NUM_LENSES):
            x_pos = self.WIDTH * (i + 1.5) / (self.NUM_LENSES + 2)
            y_pos = self.HEIGHT / 2 + self.np_random.uniform(-self.HEIGHT/4, self.HEIGHT/4)
            self.lenses.append({
                'pos': np.array([x_pos, y_pos]),
                'angle': self.np_random.uniform(0, 360),
                'width': 10,
                'height': 60
            })
        
        self._update_game_state()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_action(action)
        self.steps += 1
        
        self.last_target_intensity = self.target_intensity
        self._update_game_state()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = False # This env does not truncate based on time limit, it terminates.
        
        if terminated:
            self.game_over = True
            if self.target_intensity >= self.WIN_INTENSITY:
                reward += 100.0 # Win bonus
            else:
                reward -= 10.0 # Loss penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.action_cooldown > 0:
            self.action_cooldown -=1
        
        # --- Lens Selection ---
        if self.action_cooldown == 0:
            if movement == 3:  # Left
                self.selected_lens_idx = (self.selected_lens_idx - 1) % self.NUM_LENSES
                self.action_cooldown = 5 # Cooldown to prevent rapid cycling
            elif movement == 4:  # Right
                self.selected_lens_idx = (self.selected_lens_idx + 1) % self.NUM_LENSES
                self.action_cooldown = 5

        # --- Lens Rotation ---
        lens = self.lenses[self.selected_lens_idx]
        rotation_delta = 0.0
        
        # Coarse rotation
        if movement == 1: rotation_delta += 2.0  # CW
        elif movement == 2: rotation_delta -= 2.0  # CCW
        
        # Fine rotation
        if space_held: rotation_delta += 0.5  # Fine CW
        if shift_held: rotation_delta -= 0.5  # Fine CCW
        
        if rotation_delta != 0.0:
            lens['angle'] = (lens['angle'] + rotation_delta) % 360

    def _update_game_state(self):
        self.beam_paths, raw_intensity = self._calculate_beam_paths()
        self.target_intensity = min(1.0, raw_intensity / self.NUM_BEAM_SOURCES)

    def _calculate_beam_paths(self):
        all_paths = []
        total_intensity = 0.0

        for source in self.beam_sources:
            beam_origin = source['pos'].copy()
            beam_angle_rad = math.radians(source['angle'])
            beam_dir = np.array([math.cos(beam_angle_rad), math.sin(beam_angle_rad)])
            beam_intensity = 1.0

            for _ in range(self.MAX_BOUNCES):
                intersections = []

                # Check for intersection with target
                hit_target, dist_target = self._ray_circle_intersection(beam_origin, beam_dir, self.target['pos'], self.target['radius'])
                if hit_target:
                    intersections.append({'type': 'target', 'dist': dist_target, 'point': beam_origin + dist_target * beam_dir})

                # Check for intersections with lenses
                for i, lens in enumerate(self.lenses):
                    corners = self._get_lens_corners(lens)
                    for j in range(4):
                        p1 = corners[j]
                        p2 = corners[(j + 1) % 4]
                        hit_lens, dist_lens = self._ray_segment_intersection(beam_origin, beam_dir, p1, p2)
                        if hit_lens:
                            intersections.append({'type': 'lens', 'dist': dist_lens, 'point': beam_origin + dist_lens * beam_dir, 'lens_idx': i})
                
                if not intersections:
                    # Beam goes off-screen
                    end_point = beam_origin + 2000 * beam_dir # Far away point
                    all_paths.append({'p1': beam_origin, 'p2': end_point, 'intensity': beam_intensity})
                    break

                # Find the closest intersection
                closest_hit = min(intersections, key=lambda x: x['dist'])
                
                # Add path segment to the hit point
                all_paths.append({'p1': beam_origin, 'p2': closest_hit['point'], 'intensity': beam_intensity})
                
                if closest_hit['type'] == 'target':
                    total_intensity += beam_intensity
                    break
                
                if closest_hit['type'] == 'lens':
                    # Refract the beam
                    beam_origin = closest_hit['point']
                    beam_intensity *= self.LENS_INTENSITY_LOSS
                    
                    hit_lens = self.lenses[closest_hit['lens_idx']]
                    new_angle_rad = math.radians(hit_lens['angle'])
                    beam_dir = np.array([math.cos(new_angle_rad), math.sin(new_angle_rad)])
        
        return all_paths, total_intensity

    def _calculate_reward(self):
        reward = -0.01  # Step penalty

        intensity_delta = self.target_intensity - self.last_target_intensity
        if intensity_delta > 0:
            reward += intensity_delta * 10.0  # +0.1 per 1% increase

        if self.target_intensity >= self.WIN_INTENSITY and not self.win_event_triggered:
            reward += 5.0
            self.win_event_triggered = True
        
        return reward

    def _check_termination(self):
        if self.target_intensity >= self.WIN_INTENSITY:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Target
        target_color = self.COLOR_TARGET_WIN if self.target_intensity >= self.WIN_INTENSITY else self.COLOR_TARGET
        glow_radius = int(self.target['radius'] * (1 + self.target_intensity * 0.5))
        glow_alpha = int(50 * self.target_intensity)
        if glow_alpha > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(self.target['pos'][0]), int(self.target['pos'][1]), glow_radius, (*target_color, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, int(self.target['pos'][0]), int(self.target['pos'][1]), int(self.target['radius']), target_color)

        # Render Beam Sources
        for source in self.beam_sources:
            pygame.gfxdraw.filled_circle(self.screen, int(source['pos'][0]), int(source['pos'][1]), 8, self.COLOR_BEAM)
            pygame.gfxdraw.aacircle(self.screen, int(source['pos'][0]), int(source['pos'][1]), 8, self.COLOR_BEAM)

        # Render Beams
        for path in self.beam_paths:
            self._render_beam(self.screen, path['p1'], path['p2'], path['intensity'])
        
        # Render Lenses
        for i, lens in enumerate(self.lenses):
            is_selected = (i == self.selected_lens_idx)
            self._render_lens(self.screen, lens, is_selected)
    
    def _render_beam(self, surf, p1, p2, intensity):
        if intensity < 0.01: return
        
        alpha = int(max(0, min(255, 200 * intensity)))
        color = (*self.COLOR_BEAM, alpha)
        
        # Glow effect by drawing multiple lines
        pygame.draw.aaline(surf, color, p1, p2, blend=True)
        if intensity > 0.2:
             # Using line instead of aaline for thickness
             pygame.draw.line(surf, (*self.COLOR_BEAM, int(alpha/4)), p1, p2, width=5)
             pygame.draw.line(surf, (*self.COLOR_BEAM, int(alpha/8)), p1, p2, width=9)

    def _render_lens(self, surf, lens, is_selected):
        corners = self._get_lens_corners(lens)
        int_corners = [(int(p[0]), int(p[1])) for p in corners]
        
        # Draw transparent filled polygon
        pygame.gfxdraw.filled_polygon(surf, int_corners, (*self.COLOR_LENS, 80))
        
        # Draw anti-aliased outline
        outline_color = self.COLOR_LENS_SELECTED if is_selected else self.COLOR_LENS
        pygame.gfxdraw.aapolygon(surf, int_corners, outline_color)
        if is_selected:
             # Thicker outline for selected lens
             pygame.draw.polygon(surf, outline_color, int_corners, 2)

    def _render_ui(self):
        # Render Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Render Intensity in Target
        intensity_pct = self.target_intensity * 100
        intensity_text = f"{intensity_pct:.0f}%"
        intensity_color = self.COLOR_TARGET_WIN if self.target_intensity >= self.WIN_INTENSITY else self.COLOR_TARGET
        intensity_surf = self.font_large.render(intensity_text, True, intensity_color)
        text_rect = intensity_surf.get_rect(center=(self.target['pos'][0], self.target['pos'][1]))
        self.screen.blit(intensity_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "target_intensity": self.target_intensity,
            "selected_lens": self.selected_lens_idx,
        }

    # --- Helper Functions ---
    @staticmethod
    def _get_lens_corners(lens):
        w, h = lens['width'] / 2, lens['height'] / 2
        angle_rad = math.radians(lens['angle'])
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        
        corners = [np.array([-w, -h]), np.array([w, -h]), np.array([w, h]), np.array([-w, h])]
        rot_matrix = np.array([[c, -s], [s, c]])
        
        return [lens['pos'] + rot_matrix @ p for p in corners]

    @staticmethod
    def _ray_circle_intersection(ray_origin, ray_dir, circle_center, radius):
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False, -1
        else:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2.0 * a)
            t2 = (-b + sqrt_d) / (2.0 * a)
            if t1 >= 1e-4: return True, t1 # Use a small epsilon
            if t2 >= 1e-4: return True, t2
            return False, -1

    @staticmethod
    def _ray_segment_intersection(ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = np.array([-ray_dir[1], ray_dir[0]])
        dot = np.dot(v2, v3)
        if abs(dot) < 1e-6:
            return False, -1 # Parallel lines

        t1 = np.cross(v2, v1) / dot
        t2 = np.dot(v1, v3) / dot

        if t1 >= 1e-4 and 0 <= t2 <= 1:
            return True, t1
        return False, -1

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Light Bender")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------\n")

    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1      # Coarse CW
        if keys[pygame.K_DOWN]: movement = 2    # Coarse CCW
        if keys[pygame.K_LEFT]: movement = 3    # Select prev
        if keys[pygame.K_RIGHT]: movement = 4   # Select next
        if keys[pygame.K_SPACE]: space = 1      # Fine CW
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1 # Fine CCW
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Intensity: {info['target_intensity']*100:.1f}%")
            pygame.time.wait(3000)

    env.close()