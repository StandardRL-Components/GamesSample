import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch a sphere onto a procedurally generated track. Navigate twists, turns, and loops "
        "to reach all the checkpoints before you fall off."
    )
    user_guide = (
        "Controls: When stopped, use ↑/↓ to adjust power and ←/→ to aim. Press space to launch. "
        "The sphere will then roll on its own."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_TRACK = (100, 200, 255)
    COLOR_SPHERE = (255, 0, 128)
    COLOR_CHECKPOINT = (255, 215, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_LAUNCH_VECTOR = (0, 255, 128)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 1500
    NUM_CHECKPOINTS = 3
    GRAVITY = 0.03
    FRICTION = 0.995
    TRACK_WIDTH = 8.0
    SPHERE_RADIUS = 5.0
    LAUNCH_POWER_INCREMENT = 0.1
    LAUNCH_ANGLE_INCREMENT = 0.1
    MAX_LAUNCH_POWER = 3.0
    MIN_LAUNCH_POWER = 0.5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sphere_pos = None
        self.sphere_vel = None
        self.is_stopped = True
        self.prev_speed = 0
        self.track_points = []
        self.checkpoints = []
        self.checkpoints_reached = 0
        self.launch_power = 0
        self.launch_angle = 0
        self.camera_pos = None
        self.particles = deque(maxlen=100)
        self.difficulty = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.difficulty = 0

        self.sphere_pos = np.array([0.0, self.SPHERE_RADIUS, 10.0])
        self.sphere_vel = np.array([0.0, 0.0, 0.0])
        self.is_stopped = True
        self.prev_speed = 0

        self.launch_power = (self.MAX_LAUNCH_POWER + self.MIN_LAUNCH_POWER) / 2
        self.launch_angle = 0

        self.camera_pos = np.array([0.0, 50.0, -40.0])
        self.particles.clear()

        self._generate_initial_track()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._update_player_controls(movement, space_held)
        self._update_physics()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self.game_over or self.checkpoints_reached >= self.NUM_CHECKPOINTS
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.checkpoints_reached >= self.NUM_CHECKPOINTS:  # Win condition
            reward += 50
            self.score += 50
        elif terminated and self.game_over:  # Lose condition
            reward -= 100
            self.score -= 100
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_controls(self, movement, space_held):
        if self.is_stopped:
            # 1=up, 2=down, 3=left, 4=right
            if movement == 1: # Increase power
                self.launch_power = min(self.MAX_LAUNCH_POWER, self.launch_power + self.LAUNCH_POWER_INCREMENT)
            elif movement == 2: # Decrease power
                self.launch_power = max(self.MIN_LAUNCH_POWER, self.launch_power - self.LAUNCH_POWER_INCREMENT)
            elif movement == 3: # Rotate left
                self.launch_angle -= self.LAUNCH_ANGLE_INCREMENT
            elif movement == 4: # Rotate right
                self.launch_angle += self.LAUNCH_ANGLE_INCREMENT

            if space_held:
                # Launch the sphere
                self.is_stopped = False
                launch_vx = self.launch_power * math.sin(self.launch_angle)
                launch_vz = self.launch_power * math.cos(self.launch_angle)
                self.sphere_vel = np.array([launch_vx, 1.0, launch_vz]) # Add a little upward pop

    def _update_physics(self):
        if self.is_stopped:
            self.sphere_vel = np.zeros(3)
            return

        # Store previous speed for reward calculation
        self.prev_speed = np.linalg.norm(self.sphere_vel)

        # Apply gravity
        self.sphere_vel[1] -= self.GRAVITY
        
        # Update position
        self.sphere_pos += self.sphere_vel

        # Track interaction
        on_track, segment_idx, turn_penalty, loop_bonus = self._check_track_collision()
        
        if on_track:
            self.sphere_vel *= self.FRICTION
            self.sphere_vel *= (1.0 - turn_penalty) # Apply turn penalty
            self.sphere_vel *= (1.0 + loop_bonus) # Apply loop bonus
        else:
            if self.sphere_pos[1] < -50: # Fell into the void
                self.game_over = True
        
        # Update camera to follow sphere
        target_cam_pos = self.sphere_pos + np.array([0, 40.0, -60.0])
        self.camera_pos = self.camera_pos * 0.9 + target_cam_pos * 0.1

        # Add particle for trail
        self.particles.append({'pos': self.sphere_pos.copy(), 'alpha': 255})
        
        # Check if sphere has stopped
        if np.linalg.norm(self.sphere_vel) < 0.1 and on_track:
            self.is_stopped = True
            self.sphere_vel = np.zeros(3)
            self.launch_angle = self._get_current_track_angle(segment_idx)

    def _check_track_collision(self):
        if len(self.track_points) < 2:
            return False, -1, 0, 0

        # Find closest track segment
        min_dist_sq = float('inf')
        closest_segment_idx = -1
        
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]['pos']
            p2 = self.track_points[i+1]['pos']
            # Simple check to only consider segments near the sphere's Z
            if not (min(p1[2], p2[2]) - 20 < self.sphere_pos[2] < max(p1[2], p2[2]) + 20):
                continue

            # Vector math to find closest point on line segment
            line_vec = p2 - p1
            point_vec = self.sphere_pos - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue
            
            t = np.dot(point_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_sq = np.sum((self.sphere_pos - closest_point)**2)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_segment_idx = i

        if closest_segment_idx == -1:
            return False, -1, 0, 0
        
        # Check if we are on the track
        dist = math.sqrt(min_dist_sq)
        if dist < self.TRACK_WIDTH:
            p1 = self.track_points[closest_segment_idx]['pos']
            p2 = self.track_points[closest_segment_idx+1]['pos']
            
            # Correct sphere's height
            line_vec = p2 - p1
            point_vec = self.sphere_pos - p1
            line_len_sq = np.dot(line_vec, line_vec)
            t = np.dot(point_vec[:2], line_vec[:2]) / np.dot(line_vec[:2], line_vec[:2]) if np.dot(line_vec[:2], line_vec[:2]) > 0 else 0
            t = np.clip(t, 0, 1)
            track_y_at_sphere = p1[1] + t * (p2[1] - p1[1])
            self.sphere_pos[1] = track_y_at_sphere + self.SPHERE_RADIUS

            # Steer velocity along track
            track_dir = (p2 - p1) / np.linalg.norm(p2 - p1)
            speed = np.linalg.norm(self.sphere_vel)
            self.sphere_vel = track_dir * speed
            
            # Calculate turn penalty
            turn_penalty = 0
            if closest_segment_idx > 0:
                p0 = self.track_points[closest_segment_idx-1]['pos']
                prev_dir = (p1-p0) / np.linalg.norm(p1-p0)
                angle = math.acos(np.clip(np.dot(track_dir, prev_dir), -1.0, 1.0))
                if angle > 0.3: # Sharp turn
                    turn_penalty = 0.30

            # Check for loop bonus
            loop_bonus = 0
            if self.track_points[closest_segment_idx].get('loop', False):
                loop_bonus = 0.20

            return True, closest_segment_idx, turn_penalty, loop_bonus
        
        return False, -1, 0, 0

    def _calculate_reward(self):
        reward = 0
        
        # Reward for moving forward
        if self.sphere_vel[2] > 0:
            reward += 0.1

        # Penalty for slowing down
        current_speed = np.linalg.norm(self.sphere_vel)
        if current_speed < self.prev_speed and not self.is_stopped:
            reward -= 0.1
        self.prev_speed = current_speed

        # Reward for checkpoints
        for i, cp in enumerate(self.checkpoints):
            if cp['active']:
                dist = np.linalg.norm(self.sphere_pos - cp['pos'])
                if dist < self.SPHERE_RADIUS + cp['radius']:
                    cp['active'] = False
                    self.checkpoints_reached += 1
                    reward += 5
                    self.difficulty += 1
                    self._extend_track()
                    break # Only one checkpoint per step
        
        return reward

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
            "checkpoints": self.checkpoints_reached,
            "speed": np.linalg.norm(self.sphere_vel),
        }

    def _project(self, pos):
        # Simple perspective projection
        pos_relative = pos - self.camera_pos
        if pos_relative[2] <= 0.1: # Behind or too close to camera
            return None, None, None

        fov = 300
        scale = fov / pos_relative[2]
        
        # Cap scale to prevent coordinates from becoming excessively large, which causes Pygame errors.
        scale = min(scale, 1000)

        x = self.WIDTH / 2 + pos_relative[0] * scale
        y = self.HEIGHT / 2 - pos_relative[1] * scale
        return int(x), int(y), scale

    def _render_game(self):
        # --- Collect and sort all drawable objects by Z ---
        render_queue = []

        # Add track segments
        for i in range(len(self.track_points) - 1):
            p1, p2 = self.track_points[i]['pos'], self.track_points[i+1]['pos']
            z_avg = (p1[2] + p2[2]) / 2
            render_queue.append({'type': 'track', 'p1': p1, 'p2': p2, 'z': z_avg})

        # Add checkpoints
        for cp in self.checkpoints:
            render_queue.append({'type': 'checkpoint', 'data': cp, 'z': cp['pos'][2]})
        
        # Add particles
        for p in self.particles:
            render_queue.append({'type': 'particle', 'data': p, 'z': p['pos'][2]})

        # Add sphere
        render_queue.append({'type': 'sphere', 'z': self.sphere_pos[2]})
        
        # Add launch vector if stopped
        if self.is_stopped:
            render_queue.append({'type': 'launch_vector', 'z': self.sphere_pos[2] - 1})

        # Sort by Z-depth (descending)
        render_queue.sort(key=lambda item: item['z'], reverse=True)

        # --- Render everything in order ---
        for item in render_queue:
            if item['type'] == 'track':
                x1, y1, s1 = self._project(item['p1'])
                x2, y2, s2 = self._project(item['p2'])
                if x1 is not None and x2 is not None:
                    width = int(max(1, self.TRACK_WIDTH * s1 * 0.5))
                    pygame.draw.line(self.screen, self.COLOR_TRACK, (x1, y1), (x2, y2), width)
            
            elif item['type'] == 'checkpoint':
                cp = item['data']
                x, y, scale = self._project(cp['pos'])
                if x is not None and cp['active']:
                    radius = int(cp['radius'] * scale)
                    if radius > 0:
                        # Glow effect
                        for i in range(radius, 0, -2):
                            alpha = int(150 * (i / radius)**2)
                            color = (*self.COLOR_CHECKPOINT, alpha)
                            try:
                                pygame.gfxdraw.filled_circle(self.screen, x, y, i, color)
                            except OverflowError: # Catch rare overflow if coords are still too big
                                pass
                        try:
                            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_CHECKPOINT)
                        except OverflowError:
                            pass

            elif item['type'] == 'particle':
                p = item['data']
                x, y, scale = self._project(p['pos'])
                if x is not None:
                    radius = int(max(1, self.SPHERE_RADIUS * scale * 0.5 * (p['alpha'] / 255)))
                    if radius > 0:
                        color = (*self.COLOR_SPHERE, int(p['alpha'] * 0.5))
                        try:
                            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
                        except OverflowError:
                            pass
                p['alpha'] = max(0, p['alpha'] - 5)

            elif item['type'] == 'sphere':
                x, y, scale = self._project(self.sphere_pos)
                if x is not None:
                    radius = int(max(1, self.SPHERE_RADIUS * scale))
                    try:
                        # Glow effect
                        for i in range(radius + 5, radius - 1, -1):
                            alpha = int(80 * (1 - (radius + 5 - i) / 6)**2)
                            color = (*self.COLOR_SPHERE, alpha)
                            pygame.gfxdraw.filled_circle(self.screen, x, y, i, color)
                        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_SPHERE)
                        pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_SPHERE)
                    except OverflowError:
                        pass
            
            elif item['type'] == 'launch_vector':
                x, y, scale = self._project(self.sphere_pos)
                if x is not None:
                    end_x = x + math.sin(self.launch_angle) * self.launch_power * 20 * scale
                    end_y = y - math.cos(self.launch_angle) * self.launch_power * 20 * scale
                    try:
                        pygame.draw.line(self.screen, self.COLOR_LAUNCH_VECTOR, (x, y), (int(end_x), int(end_y)), 2)
                    except OverflowError:
                        pass

    def _render_ui(self):
        # Checkpoints
        cp_text = f"CHECKPOINTS: {self.checkpoints_reached} / {self.NUM_CHECKPOINTS}"
        text_surface = self.font_large.render(cp_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Speed
        speed = np.linalg.norm(self.sphere_vel)
        speed_text = f"SPEED: {speed:.1f}"
        text_surface = self.font_small.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 40))

        # Game Over / Win Text
        msg = None
        if self.game_over:
            msg = "FELL OFF THE TRACK"
            color = (255, 50, 50)
        elif self.checkpoints_reached >= self.NUM_CHECKPOINTS:
            msg = "ALL CHECKPOINTS REACHED!"
            color = (50, 255, 50)
        
        if msg:
            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _generate_initial_track(self):
        self.track_points = [{'pos': np.array([0.0, 0.0, 0.0]), 'loop': False}]
        self.checkpoints = []
        
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for _ in range(30): # Initial straight section
            current_pos = current_pos + np.array([0.0, 0.0, 10.0])
            self.track_points.append({'pos': current_pos.copy(), 'loop': False})
        
        self._extend_track() # Generate first section and checkpoint

    def _extend_track(self):
        if len(self.checkpoints) >= self.NUM_CHECKPOINTS:
            return

        current_pos = self.track_points[-1]['pos']
        current_angle = self._get_current_track_angle(len(self.track_points) - 2)
        
        turn_sharpness = 0.05 + self.difficulty * 0.02
        slope_variation = 5.0 + self.difficulty * 2.0
        
        for _ in range(self.np_random.integers(40, 61)):
            current_angle += self.np_random.uniform(-turn_sharpness, turn_sharpness)
            dx = 10 * math.sin(current_angle)
            dz = 10 * math.cos(current_angle)
            dy = self.np_random.uniform(-slope_variation, slope_variation)
            
            current_pos = current_pos + np.array([dx, 0, dz])
            current_pos[1] += dy
            
            # Clamp height to prevent extreme tracks
            current_pos[1] = np.clip(current_pos[1], -100, 200)

            self.track_points.append({'pos': current_pos.copy(), 'loop': False})

        # Add a checkpoint
        cp_pos = current_pos + np.array([0, 15.0, 0])
        self.checkpoints.append({'pos': cp_pos, 'active': True, 'radius': 10.0})

    def _get_current_track_angle(self, segment_idx):
        if segment_idx < 0 or segment_idx >= len(self.track_points) - 1:
            return 0.0
        p1 = self.track_points[segment_idx]['pos']
        p2 = self.track_points[segment_idx+1]['pos']
        dx, _, dz = p2 - p1
        return math.atan2(dx, dz)

if __name__ == '__main__':
    # This block will not run in the hosted environment but is useful for local testing.
    # To run, you'll need to `pip install pygame`.
    
    # Re-enable the display for local testing.
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for display
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Momentum Sphere")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0] # Shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    pygame.quit()