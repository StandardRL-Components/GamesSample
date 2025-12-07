import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:52.746707
# Source Brief: brief_00716.md
# Brief Index: 716
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Guide a fading light beam through a maze of player-placed mirrors
    to reach a target. This environment prioritizes visual quality and engaging,
    real-time puzzle mechanics.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement): Rotate the held mirror.
      - 0: None
      - 1: Clockwise (slow)
      - 2: Counter-clockwise (slow)
      - 3: Clockwise (fast)
      - 4: Counter-clockwise (fast)
    - action[1] (Space): Place the held mirror. (1 = pressed)
    - action[2] (Shift): Acquire a new mirror to place. (1 = pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +100 for hitting the target.
    - -100 for running out of time or beam intensity.
    - +5 for placing a mirror that brings the beam closer to the target.
    - -1 for attempting to place a mirror out of bounds.
    - Continuous small reward/penalty based on the beam's change in distance
      to the target each step.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a fading light beam to a target by placing and rotating mirrors in its path."
    user_guide = "Controls: Use arrow keys to rotate the held mirror (↑/← for clockwise, ↓/→ for counter-clockwise). Press Shift to get a new mirror and Space to place it."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_BEAM = (255, 255, 255)
    COLOR_BEAM_GLOW = (200, 220, 255)
    COLOR_MIRROR = (100, 200, 255)
    COLOR_MIRROR_GHOST = (100, 200, 255)
    COLOR_TARGET = (80, 255, 150)
    COLOR_TARGET_GLOW = (80, 255, 150)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_UI_BAR_BG = (50, 60, 70)
    COLOR_UI_BAR_FG = (255, 255, 255)

    # Game parameters
    BEAM_INTENSITY_DECAY_PER_SEC = 2.0
    BEAM_MAX_REFLECTIONS = 15
    MIRROR_SIZE = (80, 5)
    MIRROR_ROT_SPEED_SLOW = 1.5
    MIRROR_ROT_SPEED_FAST = 5.0
    TARGET_RADIUS = 20

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.render_mode = render_mode
        
        self.placed_mirrors = []
        self.held_mirror = None
        self.beam_origin = (0, 0)
        self.beam_initial_angle = 0
        self.beam_segments = []
        self.beam_intensity = 100.0
        self.target_pos = (0, 0)
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_beam_end_pos = (0,0)
        self.last_dist_to_target = float('inf')
        self.event_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.beam_origin = (30, self.HEIGHT // 2)
        self.beam_initial_angle = self.np_random.uniform(-15, 15)
        self.beam_intensity = 100.0
        
        self.target_pos = (
            self.np_random.integers(self.WIDTH * 0.75, self.WIDTH - 40),
            self.np_random.integers(40, self.HEIGHT - 40)
        )
        
        self.placed_mirrors = []
        self.held_mirror = None
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False
        self.event_reward = 0.0

        self._update_beam_path()
        self.last_dist_to_target = self._get_beam_dist_to_target()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        self._handle_actions(action)
        
        # --- Update game state ---
        self.beam_intensity -= self.BEAM_INTENSITY_DECAY_PER_SEC / self.FPS
        
        dist_before_update = self._get_beam_dist_to_target()
        
        victory = self._update_beam_path()
        
        dist_after_update = self._get_beam_dist_to_target()

        # --- Calculate Rewards ---
        # Continuous distance-based reward
        dist_change = dist_before_update - dist_after_update
        if abs(dist_change) > 1e-6:
             reward += np.clip(dist_change * 0.01, -0.1, 0.1)

        # Event-based rewards are handled in _handle_actions
        reward += self.event_reward
        self.event_reward = 0.0 # Reset event reward

        # Terminal rewards
        terminated = self._check_termination()
        if victory:
            reward += 100.0
        elif terminated and not victory:
            reward -= 100.0

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # Action: Pick up a new mirror
        if shift_pressed and self.held_mirror is None:
            # sfx: new_mirror_pickup
            self.held_mirror = {
                "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2]),
                "angle": 0,
                "size": self.MIRROR_SIZE
            }

        # Action: Rotate held mirror
        if self.held_mirror is not None:
            rot = 0
            if movement == 1: rot = -self.MIRROR_ROT_SPEED_SLOW   # Up -> CW
            elif movement == 2: rot = self.MIRROR_ROT_SPEED_SLOW  # Down -> CCW
            elif movement == 3: rot = -self.MIRROR_ROT_SPEED_FAST # Left -> CW Large
            elif movement == 4: rot = self.MIRROR_ROT_SPEED_FAST  # Right -> CCW Large
            self.held_mirror["angle"] = (self.held_mirror["angle"] + rot) % 360

        # Action: Place held mirror
        if space_pressed and self.held_mirror is not None:
            # sfx: place_mirror
            dist_before_place = self._get_beam_dist_to_target()
            
            if 0 < self.held_mirror['pos'][0] < self.WIDTH and \
               0 < self.held_mirror['pos'][1] < self.HEIGHT:
                self.placed_mirrors.append(self.held_mirror.copy())
                self.held_mirror = None

                self._update_beam_path()
                dist_after_place = self._get_beam_dist_to_target()
                if dist_after_place < dist_before_place:
                    self.event_reward += 5.0
            else:
                # sfx: invalid_action
                self.event_reward -= 1.0

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

    def _update_beam_path(self):
        self.beam_segments = []
        
        current_pos = np.array(self.beam_origin, dtype=float)
        current_angle_rad = math.radians(self.beam_initial_angle)
        
        for _ in range(self.BEAM_MAX_REFLECTIONS + 1):
            ray_dir = np.array([math.cos(current_angle_rad), math.sin(current_angle_rad)])
            
            intersections = []
            
            # Check mirror intersections
            for i, mirror in enumerate(self.placed_mirrors):
                intersect_info = self._get_ray_mirror_intersection(current_pos, ray_dir, mirror)
                if intersect_info:
                    intersect_info["type"] = "mirror"
                    intersect_info["mirror_idx"] = i
                    intersections.append(intersect_info)

            # Check boundary intersections
            if ray_dir[0] != 0:
                t_right = (self.WIDTH - current_pos[0]) / ray_dir[0]
                if t_right > 1e-6: intersections.append({"t": t_right, "pos": current_pos + t_right * ray_dir, "type": "wall"})
                t_left = -current_pos[0] / ray_dir[0]
                if t_left > 1e-6: intersections.append({"t": t_left, "pos": current_pos + t_left * ray_dir, "type": "wall"})
            if ray_dir[1] != 0:
                t_bottom = (self.HEIGHT - current_pos[1]) / ray_dir[1]
                if t_bottom > 1e-6: intersections.append({"t": t_bottom, "pos": current_pos + t_bottom * ray_dir, "type": "wall"})
                t_top = -current_pos[1] / ray_dir[1]
                if t_top > 1e-6: intersections.append({"t": t_top, "pos": current_pos + t_top * ray_dir, "type": "wall"})

            if not intersections: break 

            closest_hit = min(intersections, key=lambda x: x["t"])
            
            self.beam_segments.append((current_pos.copy(), closest_hit["pos"]))
            self.last_beam_end_pos = closest_hit["pos"]

            if self._check_target_hit(current_pos, closest_hit["pos"]):
                self.game_over = True
                return True # Victory

            if closest_hit["type"] == "mirror":
                # sfx: beam_reflect
                current_pos = closest_hit["pos"]
                mirror = self.placed_mirrors[closest_hit["mirror_idx"]]
                mirror_angle_rad = math.radians(mirror["angle"])
                normal = np.array([-math.sin(mirror_angle_rad), math.cos(mirror_angle_rad)])
                
                if np.dot(ray_dir, normal) > 0: normal = -normal

                reflected_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
                current_angle_rad = math.atan2(reflected_dir[1], reflected_dir[0])
            else: break
        
        return False

    def _get_ray_mirror_intersection(self, ray_origin, ray_dir, mirror):
        w, h = mirror["size"]
        angle_rad = math.radians(mirror["angle"])
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        p1 = mirror["pos"] + np.array([-w/2*cos_a, -w/2*sin_a])
        p2 = mirror["pos"] + np.array([w/2*cos_a, w/2*sin_a])
        
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = np.array([-ray_dir[1], ray_dir[0]])
        
        dot_v2_v3 = np.dot(v2, v3)
        if abs(dot_v2_v3) < 1e-6: return None

        t1 = np.cross(v2, v1) / dot_v2_v3
        t2 = np.dot(v1, v3) / dot_v2_v3

        if t1 >= 1e-6 and 0 <= t2 <= 1:
            return {"t": t1, "pos": ray_origin + t1 * ray_dir}
        return None

    def _check_target_hit(self, seg_start, seg_end):
        p = np.array(self.target_pos)
        a = seg_start
        b = seg_end
        
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0: return np.linalg.norm(p - a) <= self.TARGET_RADIUS

        proj = np.clip(np.dot(p - a, ab) / ab_squared, 0, 1)
        closest_point = a + proj * ab
        
        return np.linalg.norm(p - closest_point) <= self.TARGET_RADIUS

    def _get_beam_dist_to_target(self):
        if not self.beam_segments:
            return np.linalg.norm(np.array(self.beam_origin) - np.array(self.target_pos))
        
        end_pos = self.beam_segments[-1][1]
        return np.linalg.norm(end_pos - np.array(self.target_pos))

    def _check_termination(self):
        return self.game_over or self.beam_intensity <= 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "beam_intensity": self.beam_intensity,
            "mirrors_placed": len(self.placed_mirrors),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_glowing_circle(self.screen, self.COLOR_TARGET, self.COLOR_TARGET_GLOW, self.target_pos, self.TARGET_RADIUS, 15)

        for mirror in self.placed_mirrors:
            self._draw_rotated_rect(self.screen, self.COLOR_MIRROR, mirror["pos"], mirror["size"], mirror["angle"])

        self._update_and_draw_beam_and_particles()

        if self.held_mirror:
            self._draw_rotated_rect(self.screen, self.COLOR_MIRROR_GHOST, self.held_mirror["pos"], self.held_mirror["size"], self.held_mirror["angle"], alpha=128)

        self._update_and_draw_particles()

    def _update_and_draw_beam_and_particles(self):
        beam_alpha = int(200 * (max(0, self.beam_intensity) / 100.0) + 55)
        
        for start, end in self.beam_segments:
            if self.np_random.random() < 0.7: self._spawn_particles(start, end, 1)
            pygame.draw.aaline(self.screen, self.COLOR_BEAM_GLOW + (int(beam_alpha * 0.3),), start, end, 5)
            pygame.draw.aaline(self.screen, self.COLOR_BEAM + (beam_alpha,), start, end, 2)

    def _spawn_particles(self, p1, p2, num):
        for _ in range(num):
            pos = p1 + (p2 - p1) * self.np_random.random()
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.2, 0.8)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 40)
            self.particles.append({"pos": list(pos), "vel": vel, "life": lifetime, "max_life": lifetime})

    def _update_and_draw_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

            if p["life"] <= 0:
                self.particles.pop(i)
            else:
                life_ratio = p["life"] / p["max_life"]
                alpha = int(life_ratio * 150)
                radius = life_ratio * 2
                color = self.COLOR_BEAM_GLOW + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(radius), color)

    def _render_ui(self):
        bar_width, bar_height = 200, 15
        bar_x, bar_y = 10, 10
        intensity_ratio = np.clip(self.beam_intensity / 100.0, 0, 1)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, int(bar_width * intensity_ratio), bar_height))
        intensity_text = self.font_small.render(f"BEAM: {int(self.beam_intensity)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(intensity_text, (bar_x + bar_width + 10, bar_y))
        
        time_left = self.MAX_TIME_SECONDS - (self.steps / self.FPS)
        time_text = self.font_large.render(f"{max(0, time_left):.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

    def _draw_rotated_rect(self, surface, color, center, size, angle, alpha=255):
        w, h = size
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = center[0] + (i * w/2 * cos_a - j * h/2 * sin_a)
                y = center[1] + (i * w/2 * sin_a + j * h/2 * cos_a)
                corners.append((x, y))
        
        corners = [corners[0], corners[1], corners[3], corners[2]]
        
        if alpha < 255:
            surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(surf, corners, color + (alpha,))
            pygame.gfxdraw.aapolygon(surf, corners, color + (alpha,))
            surface.blit(surf, (0,0))
        else:
            pygame.gfxdraw.filled_polygon(surface, corners, color)
            pygame.gfxdraw.aapolygon(surface, corners, color)

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius, glow_strength):
        center_int = (int(center[0]), int(center[1]))
        for i in range(glow_strength, 0, -2):
            alpha = int(50 * (1 - i / glow_strength))
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius + i, glow_color + (alpha,))
        
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Light Bender Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Environment Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()