import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:45:08.812403
# Source Brief: brief_02351.md
# Brief Index: 2351
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where an agent controls a mountain goat.
    The goal is to climb a procedurally generated mountain to reach the summit,
    while avoiding patrolling wolves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Climb a treacherous mountain as a goat, avoiding patrolling wolves to reach the summit. "
        "Stand still to enter stealth mode and reduce detection."
    )
    user_guide = "Controls: ←→ to move, ↑ to jump. Hold space to climb steep walls."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH_FACTOR = 3
    WORLD_WIDTH = SCREEN_WIDTH * WORLD_WIDTH_FACTOR

    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_MOUNTAIN_FAR = (100, 120, 150)
    COLOR_BG_MOUNTAIN_NEAR = (120, 140, 170)
    COLOR_SNOW = (240, 245, 255)
    COLOR_GOAT = (139, 69, 19)
    COLOR_GOAT_STEALTH = (200, 200, 210)
    COLOR_WOLF = (80, 80, 90)
    COLOR_WOLF_DETECT = (255, 50, 50, 100)
    COLOR_PEAK_MARKER = (255, 215, 0)
    COLOR_UI_TEXT = (10, 20, 30)

    # Physics and Gameplay
    GRAVITY = 0.4
    FRICTION = 0.9
    GOAT_SPEED = 3.5
    GOAT_JUMP_POWER = -9.0
    GOAT_CLIMB_SPEED = 2.0
    CLIMBABLE_SLOPE_THRESHOLD = 2.5
    MAX_STEPS = 2000
    PEAK_HEIGHT_Y = 50 # Y-coordinate to reach for victory

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State Initialization ---
        # These will be properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.goat_pos = np.zeros(2, dtype=np.float32)
        self.goat_vel = np.zeros(2, dtype=np.float32)
        self.is_grounded = False
        self.is_climbing = False
        self.stealth_timer = 0
        self.is_stealthed = False
        
        self.terrain_points = []
        self.climbable_segments = []
        self.wolves = []
        self.particles = []
        
        self.camera_offset = np.zeros(2, dtype=np.float32)

        self.last_y_pos = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self._generate_terrain()
        self._place_goat_and_wolves()
        
        self.goat_vel = np.zeros(2, dtype=np.float32)
        self.is_grounded = False
        self.is_climbing = False
        self.stealth_timer = 0
        self.is_stealthed = False
        self.particles = []
        self.last_y_pos = self.goat_pos[1]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action & Update Goat State ---
        movement, space_held, _ = action
        space_held = bool(space_held)
        
        self._handle_input(movement, space_held)
        
        # --- 2. Update Game Logic & Physics ---
        self._update_goat_physics()
        self._update_wolves()
        self._update_particles()
        self._update_camera()

        # --- 3. Calculate Reward ---
        reward = self._calculate_reward()
        self.score += reward

        # --- 4. Check for Termination ---
        self._check_termination_conditions()
        terminated = self.game_over

        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        is_moving = False
        if movement == 3:  # Left
            self.goat_vel[0] -= self.GOAT_SPEED
            is_moving = True
        elif movement == 4:  # Right
            self.goat_vel[0] += self.GOAT_SPEED
            is_moving = True
            
        # Jumping
        if movement == 1 and self.is_grounded: # Up
            self.goat_vel[1] = self.GOAT_JUMP_POWER
            self.is_grounded = False
            self._create_particles(self.goat_pos, 20, (1.5, 2.5)) # Sound: Jump sfx
            
        # Climbing
        can_climb, _ = self._check_climbable()
        if space_held and can_climb:
            self.is_climbing = True
            self.goat_vel[1] = 0 # Nullify gravity
            self.goat_vel[1] -= self.GOAT_CLIMB_SPEED
            is_moving = True
        else:
            self.is_climbing = False

        # Stealth
        if not is_moving and self.is_grounded:
            self.stealth_timer += 1
            if self.stealth_timer > 60: # 2 seconds at 30fps
                self.is_stealthed = True
        else:
            self.stealth_timer = 0
            self.is_stealthed = False

    def _update_goat_physics(self):
        # Apply friction
        self.goat_vel[0] *= self.FRICTION
        if abs(self.goat_vel[0]) < 0.1:
            self.goat_vel[0] = 0

        # Apply gravity unless climbing
        if not self.is_climbing:
            self.goat_vel[1] += self.GRAVITY
        
        # Update position
        self.goat_pos += self.goat_vel
        
        # Terrain collision
        ground_y, ground_slope = self._get_terrain_info_at(self.goat_pos[0])
        
        if self.goat_pos[1] > ground_y:
            self.goat_pos[1] = ground_y
            if self.goat_vel[1] > 2.0: # Landing effect on hard fall
                self._create_particles(self.goat_pos, 10, (1.0, 2.0)) # Sound: Land sfx
            self.goat_vel[1] = 0
            self.is_grounded = True
        else:
            self.is_grounded = False

        # Prevent goat from sliding down very steep slopes when not moving
        if self.is_grounded and abs(self.goat_vel[0]) < 0.5 and abs(ground_slope) > 0.5:
             self.goat_vel[0] = 0

        # World boundaries
        self.goat_pos[0] = np.clip(self.goat_pos[0], 0, self.WORLD_WIDTH)

    def _update_wolves(self):
        wolf_speed_multiplier = 1.0 + (0.05 * (self.steps // 500))
        for wolf in self.wolves:
            wolf['pos'][0] += wolf['vel'] * wolf_speed_multiplier
            if wolf['pos'][0] < wolf['patrol_min_x'] or wolf['pos'][0] > wolf['patrol_max_x']:
                wolf['vel'] *= -1
            
            wolf_y, _ = self._get_terrain_info_at(wolf['pos'][0])
            wolf['pos'][1] = wolf_y

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.05)

    def _update_camera(self):
        target_cam_x = self.goat_pos[0] - self.SCREEN_WIDTH / 2
        target_cam_y = self.goat_pos[1] - self.SCREEN_HEIGHT / 2
        
        # Smooth camera movement
        self.camera_offset[0] += (target_cam_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.1

        # Clamp camera to world boundaries
        self.camera_offset[0] = np.clip(self.camera_offset[0], 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)
        # No vertical clamp to allow seeing high/low areas

    def _calculate_reward(self):
        reward = 0
        
        # Reward for vertical movement
        y_diff = self.last_y_pos - self.goat_pos[1]
        if y_diff > 0: # Moved up
            reward += 0.1 * y_diff
        elif y_diff < 0: # Moved down
            reward += 0.01 * y_diff # y_diff is negative, so this is a penalty
            
        self.last_y_pos = self.goat_pos[1]
        
        # Terminal rewards are handled in _check_termination_conditions
        return reward

    def _check_termination_conditions(self):
        # Victory condition
        if self.goat_pos[1] <= self.PEAK_HEIGHT_Y:
            self.score += 100
            self.game_over = True
            self.victory = True
            return

        # Failure condition (wolf detection)
        for wolf in self.wolves:
            dist = np.linalg.norm(self.goat_pos - wolf['pos'])
            detection_radius = wolf['detect_radius']
            if self.is_stealthed:
                detection_radius *= 0.5
            
            if dist < detection_radius:
                self.score -= 10
                self.game_over = True
                self.victory = False
                # Sound: Detection alert sfx
                return

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_SKY)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "altitude": int(self.SCREEN_HEIGHT - self.goat_pos[1]),
            "is_stealthed": self.is_stealthed
        }

    # --- Rendering Methods ---
    def _render_background(self):
        # Parallax mountains
        self._draw_parallax_layer(self.COLOR_BG_MOUNTAIN_FAR, 0.2, 200, 100, 5)
        self._draw_parallax_layer(self.COLOR_BG_MOUNTAIN_NEAR, 0.5, 150, 150, 10)

    def _render_game(self):
        # Draw terrain
        terrain_screen_points = [(p[0] - self.camera_offset[0], p[1] - self.camera_offset[1]) for p in self.terrain_points]
        if len(terrain_screen_points) > 1:
            poly_points = [(terrain_screen_points[0][0], self.SCREEN_HEIGHT*2)] + terrain_screen_points + [(terrain_screen_points[-1][0], self.SCREEN_HEIGHT*2)]
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_SNOW)
            pygame.gfxdraw.aapolygon(self.screen, poly_points, self.COLOR_SNOW)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_offset[0]), int(p['pos'][1] - self.camera_offset[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), (255, 255, 255, p['life']*5))
        
        # Draw wolves
        for wolf in self.wolves:
            pos = (wolf['pos'][0] - self.camera_offset[0], wolf['pos'][1] - self.camera_offset[1])
            
            # Detection radius
            radius = wolf['detect_radius']
            if self.is_stealthed:
                radius *= 0.5
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), self.COLOR_WOLF_DETECT)
            
            # Wolf body
            p1 = (int(pos[0] - 12), int(pos[1]))
            p2 = (int(pos[0]), int(pos[1] - 8))
            p3 = (int(pos[0] + 12), int(pos[1]))
            p4 = (int(pos[0]), int(pos[1] + 8))
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_WOLF)

        # Draw goat
        goat_color = self.COLOR_GOAT_STEALTH if self.is_stealthed else self.COLOR_GOAT
        pos = (self.goat_pos[0] - self.camera_offset[0], self.goat_pos[1] - self.camera_offset[1])
        goat_points = [
            (pos[0] - 8, pos[1]),
            (pos[0] - 6, pos[1] - 15),
            (pos[0] + 6, pos[1] - 15),
            (pos[0] + 8, pos[1])
        ]
        int_goat_points = [(int(p[0]), int(p[1])) for p in goat_points]
        pygame.gfxdraw.filled_polygon(self.screen, int_goat_points, goat_color)
        pygame.gfxdraw.aapolygon(self.screen, int_goat_points, (0,0,0,50)) # Subtle outline

        # Draw peak marker
        peak_pos = (self.WORLD_WIDTH / 2 - self.camera_offset[0], self.PEAK_HEIGHT_Y - self.camera_offset[1])
        marker_points = [
            (peak_pos[0], peak_pos[1] - 15),
            (peak_pos[0] - 10, peak_pos[1]),
            (peak_pos[0] + 10, peak_pos[1])
        ]
        int_marker_points = [(int(p[0]), int(p[1])) for p in marker_points]
        pygame.gfxdraw.filled_trigon(self.screen, *int_marker_points[0], *int_marker_points[1], *int_marker_points[2], self.COLOR_PEAK_MARKER)

    def _render_ui(self):
        # Altitude display
        altitude = int(self.SCREEN_HEIGHT * 2 - self.goat_pos[1]) # Arbitrary scale
        altitude_text = self.font_ui.render(f"ALT: {altitude}m", True, self.COLOR_UI_TEXT)
        self.screen.blit(altitude_text, (10, 10))

        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Game Over / Victory text
        if self.game_over:
            if self.victory:
                msg = "SUMMIT REACHED!"
                color = self.COLOR_PEAK_MARKER
            else:
                msg = "DETECTED!"
                color = self.COLOR_WOLF_DETECT
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    # --- Helper Methods ---
    def _generate_terrain(self):
        self.terrain_points = []
        self.climbable_segments = []
        
        y = self.SCREEN_HEIGHT * 0.9
        x = 0
        segment_length = 80

        # Start on a flat platform
        self.terrain_points.append(np.array([x, y]))
        x += segment_length * 2
        self.terrain_points.append(np.array([x, y]))

        while x < self.WORLD_WIDTH:
            prev_p = self.terrain_points[-1]
            dx = random.uniform(segment_length * 0.8, segment_length * 1.2)
            dy = random.uniform(-segment_length * 1.5, segment_length * 0.5)
            
            # Ensure path is generally upward
            dy -= 10

            next_x = prev_p[0] + dx
            next_y = np.clip(prev_p[1] + dy, self.PEAK_HEIGHT_Y + 50, self.SCREEN_HEIGHT)
            
            new_p = np.array([next_x, next_y])
            self.terrain_points.append(new_p)

            # Check if climbable
            slope = abs((new_p[1] - prev_p[1]) / (new_p[0] - prev_p[0]))
            if slope > self.CLIMBABLE_SLOPE_THRESHOLD:
                self.climbable_segments.append((len(self.terrain_points) - 2, len(self.terrain_points) - 1))
            
            x = next_x
        
        # Ensure the peak is reachable
        peak_x = self.WORLD_WIDTH / 2
        peak_idx = min(range(len(self.terrain_points)), key=lambda i: abs(self.terrain_points[i][0] - peak_x))
        self.terrain_points[peak_idx][1] = self.PEAK_HEIGHT_Y + 20 # Small platform at peak

    def _place_goat_and_wolves(self):
        # Place goat at the start
        self.goat_pos = self.terrain_points[1].copy().astype(np.float32)
        self.goat_pos[1] -= 15 # Start slightly above ground

        # Place wolves
        self.wolves = []
        num_wolves = 3
        for _ in range(num_wolves):
            # Find a relatively flat platform to patrol
            for _ in range(100): # Tries
                idx = random.randint(2, len(self.terrain_points) - 2)
                p1 = self.terrain_points[idx]
                p2 = self.terrain_points[idx+1]
                slope = abs((p2[1]-p1[1])/(p2[0]-p1[0]))
                if slope < 0.3:
                    wolf_x = (p1[0] + p2[0]) / 2
                    if abs(wolf_x - self.goat_pos[0]) > self.SCREEN_WIDTH / 2: # Don't spawn too close
                        wolf_y, _ = self._get_terrain_info_at(wolf_x)
                        self.wolves.append({
                            'pos': np.array([wolf_x, wolf_y], dtype=np.float32),
                            'vel': random.choice([-1.0, 1.0]),
                            'patrol_min_x': p1[0],
                            'patrol_max_x': p2[0],
                            'detect_radius': random.uniform(70, 100)
                        })
                        break

    def _get_terrain_info_at(self, x_pos):
        for i in range(len(self.terrain_points) - 1):
            p1 = self.terrain_points[i]
            p2 = self.terrain_points[i+1]
            if p1[0] <= x_pos < p2[0]:
                # Linear interpolation
                t = (x_pos - p1[0]) / (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                return y, slope
        # If outside range, return height of first or last point
        if x_pos < self.terrain_points[0][0]:
            return self.terrain_points[0][1], 0
        return self.terrain_points[-1][1], 0

    def _check_climbable(self):
        goat_center = self.goat_pos + np.array([0, -7.5]) # Check from center of goat
        for i_p1, i_p2 in self.climbable_segments:
            p1 = self.terrain_points[i_p1]
            p2 = self.terrain_points[i_p2]
            
            # Simple bounding box check first
            if not (min(p1[0], p2[0]) - 20 < goat_center[0] < max(p1[0], p2[0]) + 20 and \
                    min(p1[1], p2[1]) - 20 < goat_center[1] < max(p1[1], p2[1]) + 20):
                continue
            
            # Distance from point to line segment
            d = np.linalg.norm(np.cross(p2-p1, p1-goat_center)) / np.linalg.norm(p2-p1)
            if d < 15: # Close enough to the wall
                return True, (p1, p2)
        return False, None

    def _create_particles(self, pos, count, speed_range):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': random.randint(20, 40),
                'radius': random.uniform(2, 5)
            })
            
    def _draw_parallax_layer(self, color, factor, base_height, amplitude, frequency):
        points = []
        for x in range(0, self.WORLD_WIDTH + 1, 50):
            y_offset = math.sin((x + self.camera_offset[0] * factor) / self.WORLD_WIDTH * frequency) * amplitude
            y = self.SCREEN_HEIGHT - base_height + y_offset
            points.append((x - self.camera_offset[0] * factor, y))
        
        if len(points) > 1:
            screen_points = [(p[0] - self.camera_offset[0]*(1-factor), p[1] - self.camera_offset[1]*0.1) for p in points]
            poly_points = [(screen_points[0][0], self.SCREEN_HEIGHT)] + screen_points + [(screen_points[-1][0], self.SCREEN_HEIGHT)]
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, color)
    
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The main block is for human play and is not part of the Gym environment
    # It will not be executed in the test environment
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # --- Manual Play Setup ---
        pygame.display.set_caption("Goat Summit")
        human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        running = True
        
        # Key mapping for human control
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        while running:
            # --- Human Input Processing ---
            movement_action = 0 # No-op
            space_action = 0
            shift_action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            for key, move_val in key_map.items():
                if keys[key]:
                    movement_action = move_val
                    break # Prioritize one movement key
            
            if keys[pygame.K_SPACE]:
                space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1
                
            action = [movement_action, space_action, shift_action]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering for Human ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit to 30 FPS

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
                pygame.time.wait(3000) # Pause before resetting
                obs, info = env.reset()

    finally:
        env.close()