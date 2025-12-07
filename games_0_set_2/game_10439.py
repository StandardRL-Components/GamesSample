import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:29:21.492470
# Source Brief: brief_00439.md
# Brief Index: 439
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a steampunk stealth game.
    The player controls a gear, flips gravity, and sabotages factory systems
    while avoiding detection by security cameras.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "In this steampunk stealth game, control a gear to sabotage factory systems. "
        "Flip gravity to navigate and avoid detection by rotating security cameras."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move, press space to sabotage targets, "
        "and use shift to flip gravity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_BG_PIPE = (40, 45, 55)
    COLOR_PLAYER = (0, 191, 255) # Deep Sky Blue
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_CAMERA_BASE = (139, 69, 19) # Saddle Brown
    COLOR_CAMERA_LENS = (255, 255, 0) # Yellow
    COLOR_CAMERA_VISION = (255, 0, 0, 40) # Translucent Red
    COLOR_TARGET_INACTIVE = (100, 100, 100)
    COLOR_TARGET_ACTIVE = (255, 140, 0) # Dark Orange
    COLOR_TARGET_SABOTAGED = (50, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GRAVITY_INDICATOR = (0, 255, 127) # Spring Green

    # Player settings
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 6

    # Camera settings
    CAMERA_BASE_RADIUS = 15
    CAMERA_VISION_ANGLE = 60 # degrees
    CAMERA_VISION_RANGE = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_obj = pygame.font.SysFont("monospace", 16)

        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mission_number = 1
        self.player_pos = pygame.math.Vector2(0, 0)
        self.gravity = 0  # 0:Down, 1:Right, 2:Up, 3:Left
        self.cameras = []
        self.targets = []
        self.particles = []
        self.active_target_idx = -1
        self.prev_shift_state = 0
        self.gravity_flip_effect = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = 0
        self.particles = []
        self.prev_shift_state = 0
        self.gravity_flip_effect = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty per step to encourage efficiency

        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_dist_to_target = self._get_dist_to_active_target()

        # Handle player movement
        self._move_player(movement)
        
        # Handle gravity flip (on press, not hold)
        if shift_held and not self.prev_shift_state:
            self.gravity = (self.gravity + 1) % 4
            self.gravity_flip_effect = 15 # Start effect
            # sfx: GRAVITY_SHIFT_SOUND
        self.prev_shift_state = shift_held

        # Handle sabotage
        if space_held:
            sabotage_reward = self._attempt_sabotage()
            reward += sabotage_reward

        # --- Game Logic Update ---
        self._update_cameras()
        self._update_particles()
        if self.gravity_flip_effect > 0:
            self.gravity_flip_effect -= 1

        # --- Reward Calculation ---
        new_dist_to_target = self._get_dist_to_active_target()
        if self.active_target_idx != -1 and new_dist_to_target < old_dist_to_target:
            reward += 0.02 # Reward for getting closer

        # --- Termination Check ---
        terminated = False
        if self._check_detection():
            reward = -50
            terminated = True
            # sfx: DETECTION_ALARM
        elif self.active_target_idx == -1: # All targets sabotaged
            reward = 50
            terminated = True
            self.mission_number += 1
            # sfx: MISSION_COMPLETE_JINGLE
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        """Procedurally generates cameras and targets based on mission number."""
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50)
        
        # --- Targets ---
        num_targets = min(1 + self.mission_number // 2, 5)
        self.targets = []
        for i in range(num_targets):
            # Place targets in reachable areas, avoiding corners initially
            x = self.np_random.integers(100, self.SCREEN_WIDTH - 100)
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - 100)
            self.targets.append({"pos": pygame.math.Vector2(x, y), "status": "inactive"})
        
        if self.targets:
            self.targets[0]["status"] = "active"
            self.active_target_idx = 0

        # --- Cameras ---
        num_cameras = min(1 + (self.mission_number - 1), 6)
        base_rot_speed = 0.5 + (self.mission_number * 0.1)
        self.cameras = []
        for i in range(num_cameras):
            pos = pygame.math.Vector2(
                self.np_random.integers(50, self.SCREEN_WIDTH - 50),
                self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            )
            # Ensure cameras don't spawn on top of each other or the player start
            while pos.distance_to(self.player_pos) < 100 or any(pos.distance_to(c['pos']) < 50 for c in self.cameras):
                 pos.x = self.np_random.integers(50, self.SCREEN_WIDTH - 50)
                 pos.y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)

            self.cameras.append({
                "pos": pos,
                "angle": self.np_random.uniform(0, 360),
                "rot_speed": self.np_random.uniform(base_rot_speed * 0.8, base_rot_speed * 1.2) * self.np_random.choice([-1, 1])
            })

    def _move_player(self, movement_action):
        direction = pygame.math.Vector2(0, 0)
        if movement_action == 1: # Up
            direction.y = -1
        elif movement_action == 2: # Down
            direction.y = 1
        elif movement_action == 3: # Left
            direction.x = -1
        elif movement_action == 4: # Right
            direction.x = 1
        
        if direction.length() > 0:
            direction.normalize_ip()
            self.player_pos += direction * self.PLAYER_SPEED

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        
    def _attempt_sabotage(self):
        """Checks if player is near the active target and sabotages it."""
        if self.active_target_idx != -1:
            target = self.targets[self.active_target_idx]
            if self.player_pos.distance_to(target["pos"]) < self.PLAYER_RADIUS + 20:
                target["status"] = "sabotaged"
                # sfx: SABOTAGE_SUCCESS
                self._create_spark_effect(target["pos"])
                
                # Find next active target
                self.active_target_idx += 1
                if self.active_target_idx < len(self.targets):
                    self.targets[self.active_target_idx]["status"] = "active"
                else:
                    self.active_target_idx = -1 # All targets done
                return 5.0 # Reward for sabotage
        return 0.0

    def _update_cameras(self):
        for cam in self.cameras:
            cam['angle'] = (cam['angle'] + cam['rot_speed']) % 360

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_detection(self):
        """Checks if the player is detected by any camera."""
        for cam in self.cameras:
            # 1. Check if player is "downstream" based on gravity
            is_downstream = False
            if self.gravity == 0 and self.player_pos.y > cam['pos'].y: is_downstream = True # Down
            elif self.gravity == 1 and self.player_pos.x > cam['pos'].x: is_downstream = True # Right
            elif self.gravity == 2 and self.player_pos.y < cam['pos'].y: is_downstream = True # Up
            elif self.gravity == 3 and self.player_pos.x < cam['pos'].x: is_downstream = True # Left
            
            if not is_downstream:
                continue

            # 2. Check if player is within the vision cone polygon
            vision_cone_poly = self._get_vision_cone_poly(cam)
            if self._point_in_polygon(self.player_pos, vision_cone_poly):
                # 3. Final line-of-sight check (currently unobstructed)
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_game()
        self._render_ui()
        
        # Apply gravity flip visual effect
        if self.gravity_flip_effect > 0:
            alpha = int(100 * (self.gravity_flip_effect / 15))
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, alpha))
            self.screen.blit(overlay, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mission": self.mission_number,
            "targets_left": sum(1 for t in self.targets if t['status'] != 'sabotaged')
        }

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_targets()
        self._render_cameras()
        self._render_player()
        self._render_particles()

    def _render_background_details(self):
        for i in range(0, self.SCREEN_WIDTH, 80):
            pygame.draw.line(self.screen, self.COLOR_BG_PIPE, (i, 0), (i, self.SCREEN_HEIGHT), 10)
        for i in range(0, self.SCREEN_HEIGHT, 80):
            pygame.draw.line(self.screen, self.COLOR_BG_PIPE, (0, i), (self.SCREEN_WIDTH, i), 10)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 1.5)
        self.screen.blit(glow_surf, (pos[0] - self.PLAYER_RADIUS * 2, pos[1] - self.PLAYER_RADIUS * 2))
        # Main gear
        self._draw_gear(self.screen, self.COLOR_PLAYER, pos, self.PLAYER_RADIUS, 8, 4)

    def _render_cameras(self):
        for cam in self.cameras:
            pos = (int(cam['pos'].x), int(cam['pos'].y))
            # Vision cone
            poly = self._get_vision_cone_poly(cam)
            pygame.gfxdraw.aapolygon(self.screen, poly, self.COLOR_CAMERA_VISION)
            pygame.gfxdraw.filled_polygon(self.screen, poly, self.COLOR_CAMERA_VISION)
            # Camera base
            pygame.draw.circle(self.screen, self.COLOR_CAMERA_BASE, pos, self.CAMERA_BASE_RADIUS)
            pygame.draw.circle(self.screen, self.COLOR_CAMERA_LENS, pos, self.CAMERA_BASE_RADIUS // 2)

    def _render_targets(self):
        for target in self.targets:
            pos = (int(target['pos'].x), int(target['pos'].y))
            color = self.COLOR_TARGET_INACTIVE
            if target['status'] == 'active':
                color = self.COLOR_TARGET_ACTIVE
            elif target['status'] == 'sabotaged':
                color = self.COLOR_TARGET_SABOTAGED
            
            pygame.draw.circle(self.screen, color, pos, 20)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 15)
            pygame.draw.circle(self.screen, color, pos, 10)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['start_life']))
            color = p['color'] + (int(alpha),)
            size = int(max(1, p['size'] * (p['life'] / p['start_life'])))
            rect = pygame.Rect(int(p['pos'].x - size//2), int(p['pos'].y - size//2), size, size)
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        # Mission Objective
        obj_text = "OBJECTIVE: Sabotage all systems"
        if self.active_target_idx == -1:
            obj_text = "MISSION COMPLETE!"
        obj_render = self.font_obj.render(obj_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(obj_render, (10, 10))
        # Gravity Indicator
        self._draw_gravity_indicator()

    def _draw_gravity_indicator(self):
        center = (30, self.SCREEN_HEIGHT - 30)
        size = 15
        points = []
        if self.gravity == 0: # Down
            points = [(center[0], center[1] + size), (center[0] - size, center[1] - size), (center[0] + size, center[1] - size)]
        elif self.gravity == 1: # Right
            points = [(center[0] + size, center[1]), (center[0] - size, center[1] - size), (center[0] - size, center[1] + size)]
        elif self.gravity == 2: # Up
            points = [(center[0], center[1] - size), (center[0] - size, center[1] + size), (center[0] + size, center[1] + size)]
        elif self.gravity == 3: # Left
            points = [(center[0] - size, center[1]), (center[0] + size, center[1] - size), (center[0] + size, center[1] + size)]
        pygame.draw.polygon(self.screen, self.COLOR_GRAVITY_INDICATOR, points)

    # --- Helper & Utility Methods ---

    def _get_dist_to_active_target(self):
        if self.active_target_idx != -1:
            target_pos = self.targets[self.active_target_idx]['pos']
            return self.player_pos.distance_to(target_pos)
        return float('inf')

    def _create_spark_effect(self, pos):
        # sfx: SPARK_BURST
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(20, 40)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": life,
                "start_life": life,
                "color": random.choice([(255, 215, 0), (255, 165, 0), (255, 255, 255)]),
                "size": random.randint(2, 5)
            })

    def _get_vision_cone_poly(self, camera):
        """Returns a list of points for the camera's vision cone polygon."""
        pos = camera['pos']
        angle = camera['angle']
        half_fov = self.CAMERA_VISION_ANGLE / 2
        
        p1 = pos
        p2 = pos + pygame.math.Vector2(self.CAMERA_VISION_RANGE, 0).rotate(angle - half_fov)
        p3 = pos + pygame.math.Vector2(self.CAMERA_VISION_RANGE, 0).rotate(angle + half_fov)
        
        return [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]

    @staticmethod
    def _point_in_polygon(point, polygon):
        """Simple point-in-triangle test."""
        x, y = point.x, point.y
        p1, p2, p3 = polygon
        
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign((x, y), p1, p2)
        d2 = sign((x, y), p2, p3)
        d3 = sign((x, y), p3, p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    @staticmethod
    def _draw_gear(surface, color, center, radius, num_teeth, tooth_height):
        """Draws a gear shape."""
        pi2 = 2 * math.pi
        
        outer_radius = radius
        inner_radius = radius - tooth_height
        tooth_angle = pi2 / (num_teeth * 2)
        
        points = []
        for i in range(num_teeth * 2):
            angle = i * tooth_angle
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((int(x), int(y)))
        
        pygame.draw.polygon(surface, color, points)
        pygame.draw.circle(surface, color, center, int(inner_radius * 0.8))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The environment now runs headlessly.
    # To visualize, you would need to render the `rgb_array` observation
    # using a library like Pygame or Matplotlib in your training loop.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Basic test loop ---
    print("Running a short test...")
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
    
    print("Test complete.")
    env.close()

    # --- Manual Play (requires display) ---
    # To run this, comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    # at the top of the file.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        print("\nManual play is disabled in headless mode.")
        print("To enable, comment out the `os.environ` line at the top of the script.")
    else:
        env = GameEnv(render_mode="rgb_array")
        
        obs, info = env.reset()
        done = False
        
        pygame.display.set_caption("Stealth Factory")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        while not done:
            movement_action = 0 # none
            space_action = 0 # released
            shift_action = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            if keys[pygame.K_SPACE]: space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
                
            action = [movement_action, space_action, shift_action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.metadata["render_fps"])

        env.close()