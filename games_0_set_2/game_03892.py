
# Generated: 2025-08-28T00:47:17.128438
# Source Brief: brief_03892.md
# Brief Index: 3892

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to aim your next track piece. Press space to draw it. "
        "The sled will follow the track you create. Reach the green finish line!"
    )

    game_description = (
        "A physics-based arcade game where you draw the track for a sled in real-time. "
        "Create ramps for jumps to score points, but be careful not to crash! "
        "Reach the finish line before the timer runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.render_mode = render_mode

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.MAX_STEPS = 600  # Corresponds to 60 seconds at 10 steps/sec
        self.PHYSICS_SUBSTEPS = 8
        self.GRAVITY = 0.2
        self.SLED_RADIUS = 8
        self.AIM_SPEED = 10
        self.MIN_TRACK_LENGTH = 15
        self.MAX_TRACK_LENGTH = 120
        self.STATIONARY_LIMIT = 15
        self.BUMP_VELOCITY_THRESHOLD = 2.5

        # --- Colors ---
        self.COLOR_BG = (230, 230, 240)
        self.COLOR_GRID = (210, 210, 220)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_SLED_BODY = (230, 50, 50)
        self.COLOR_SLED_TOP = (255, 80, 80)
        self.COLOR_FINISH1 = (50, 200, 50)
        self.COLOR_FINISH2 = (200, 255, 200)
        self.COLOR_AIM_CURSOR = (50, 50, 200)
        self.COLOR_AIM_LINE = (100, 100, 220, 150)
        self.COLOR_PARTICLE_CRASH = (255, 150, 0)
        self.COLOR_PARTICLE_JUMP = (100, 150, 255)
        self.COLOR_TEXT = (30, 30, 50)
        
        # Will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sled_pos = None
        self.sled_vel = None
        self.track_segments = None
        self.aim_cursor = None
        self.particles = None
        self.is_airborne = None
        self.stationary_counter = None
        self.last_dist_to_finish = None
        self.finish_line_rect = None
        self.finish_line_center = None
        self.crashed = False
        self.reached_finish = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crashed = False
        self.reached_finish = False
        
        start_y = self.HEIGHT - 80
        start_platform = [
            np.array([20.0, start_y]),
            np.array([120.0, start_y])
        ]
        self.track_segments = [start_platform]

        self.sled_pos = np.array([50.0, start_y - self.SLED_RADIUS])
        self.sled_vel = np.array([2.5, 0.0]) # Initial push

        self.aim_cursor = self.track_segments[-1][1] + np.array([60.0, 0.0])
        self.particles = []
        self.is_airborne = False
        self.stationary_counter = 0

        finish_x = self.WIDTH - 40
        self.finish_line_rect = pygame.Rect(finish_x, self.HEIGHT - 150, 20, 150)
        self.finish_line_center = np.array(self.finish_line_rect.center)
        self.last_dist_to_finish = np.linalg.norm(self.finish_line_center - self.sled_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        self._handle_input(movement)

        if space_held:
            # sound: draw_line.wav
            last_point = self.track_segments[-1][1]
            vec = self.aim_cursor - last_point
            dist = np.linalg.norm(vec)

            if dist >= self.MIN_TRACK_LENGTH:
                new_point = self.aim_cursor.copy()
                if dist > self.MAX_TRACK_LENGTH:
                    new_point = last_point + (vec / dist) * self.MAX_TRACK_LENGTH
                
                self.track_segments.append([last_point, new_point])
                self.aim_cursor = new_point + np.array([60.0, 0.0])
            else:
                reward -= 0.1 # Small penalty for invalid action

        # --- Physics and Game Logic Update ---
        for _ in range(self.PHYSICS_SUBSTEPS):
            event_reward = self._update_physics()
            reward += event_reward

        # --- Progress Reward ---
        current_dist = np.linalg.norm(self.finish_line_center - self.sled_pos)
        reward += (self.last_dist_to_finish - current_dist) * 0.2 # Reward for getting closer
        self.last_dist_to_finish = current_dist

        # --- State and Termination ---
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()

        if self.reached_finish:
            reward += 100
            self.score += 100
        elif self.crashed:
            reward = -100 # Crash overrides all other rewards for the step
            self.score = -100 # And sets total score

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        if movement == 1: self.aim_cursor[1] -= self.AIM_SPEED
        elif movement == 2: self.aim_cursor[1] += self.AIM_SPEED
        elif movement == 3: self.aim_cursor[0] -= self.AIM_SPEED
        elif movement == 4: self.aim_cursor[0] += self.AIM_SPEED
        
        self.aim_cursor[0] = np.clip(self.aim_cursor[0], 0, self.WIDTH)
        self.aim_cursor[1] = np.clip(self.aim_cursor[1], 0, self.HEIGHT)
        
    def _update_physics(self):
        event_reward = 0
        
        active_segment_info = self._find_active_segment()

        if active_segment_info:
            segment, normal, point_on_line = active_segment_info
            
            if self.is_airborne: # LANDING
                # sound: land.wav
                self.is_airborne = False
                event_reward += 5.0 # Jump reward
                self._spawn_particles(self.sled_pos, 10, self.COLOR_PARTICLE_JUMP, 2)
                # Check for hard landing (bump)
                impact_vel = np.dot(self.sled_vel, normal)
                if impact_vel > self.BUMP_VELOCITY_THRESHOLD:
                    event_reward -= 1.0 # Bump penalty
                    # sound: bump.wav
            
            # Snap to track
            self.sled_pos = point_on_line - normal * self.SLED_RADIUS
            
            # Apply forces
            tangent = np.array([-normal[1], normal[0]])
            gravity_force = np.array([0, self.GRAVITY])
            accel_along_track = np.dot(gravity_force, tangent)
            
            current_speed = np.dot(self.sled_vel, tangent)
            new_speed = (current_speed + accel_along_track) * 0.995 # Friction
            self.sled_vel = tangent * new_speed

        else: # AIRBORNE
            if not self.is_airborne:
                # sound: jump.wav
                self.is_airborne = True
            
            self.sled_vel[1] += self.GRAVITY
            self.sled_vel[0] *= 0.998 # Air resistance

        # Update position
        self.sled_pos += self.sled_vel / self.PHYSICS_SUBSTEPS

        # Stationary check
        if np.linalg.norm(self.sled_vel) < 0.1:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0

        # Particle trail
        if self.steps % 2 == 0:
            self._spawn_particles(self.sled_pos, 1, self.COLOR_TRACK, 1, life=10)

        return event_reward

    def _find_active_segment(self):
        closest_dist = float('inf')
        best_segment = None
        
        for segment in self.track_segments:
            p1, p2 = segment[0], segment[1]
            line_vec = p2 - p1
            if np.linalg.norm(line_vec) == 0: continue
            
            point_vec = self.sled_pos - p1
            t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
            
            if 0 <= t <= 1: # Projection is within segment
                closest_point = p1 + t * line_vec
                dist_to_line = np.linalg.norm(self.sled_pos - closest_point)

                if dist_to_line < self.SLED_RADIUS + 5 and dist_to_line < closest_dist:
                    closest_dist = dist_to_line
                    
                    normal = np.array([line_vec[1], -line_vec[0]])
                    normal = normal / np.linalg.norm(normal)
                    # Ensure normal points upwards
                    if normal[1] > 0:
                        normal = -normal
                    
                    best_segment = (segment, normal, closest_point)
        return best_segment

    def _check_termination(self):
        # 1. Reached finish line
        if self.finish_line_rect.collidepoint(self.sled_pos[0], self.sled_pos[1]):
            self.game_over = True
            self.reached_finish = True
            # sound: win.wav
            return True

        # 2. Out of bounds
        if not (0 < self.sled_pos[0] < self.WIDTH and -50 < self.sled_pos[1] < self.HEIGHT + 50):
            self.game_over = True
            self.crashed = True
            self._spawn_particles(self.sled_pos, 30, self.COLOR_PARTICLE_CRASH, 5)
            # sound: crash.wav
            return True

        # 3. Stationary
        if self.stationary_counter > self.STATIONARY_LIMIT:
            self.game_over = True
            self.crashed = True
            return True

        # 4. Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()

        # --- Game Elements ---
        self._draw_track()
        self._draw_finish_line()
        self._update_and_draw_particles()
        self._draw_sled()
        
        # --- Aiming UI ---
        if not self.game_over:
            self._draw_aim_cursor()

        # --- UI Overlay ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
        }

    def _draw_grid(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _draw_track(self):
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 1)
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, 8)

    def _draw_finish_line(self):
        for i in range(0, self.finish_line_rect.height, 10):
            color = self.COLOR_FINISH1 if (i // 10) % 2 == 0 else self.COLOR_FINISH2
            pygame.draw.rect(self.screen, color, (self.finish_line_rect.x, self.finish_line_rect.y + i, self.finish_line_rect.width, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, self.finish_line_rect, 1)

    def _draw_sled(self):
        pos = (int(self.sled_pos[0]), int(self.sled_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED_BODY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED_BODY)
        
        # Rider on top
        rider_pos = (pos[0], pos[1] - self.SLED_RADIUS // 2)
        pygame.gfxdraw.filled_circle(self.screen, rider_pos[0], rider_pos[1], self.SLED_RADIUS // 2, self.COLOR_SLED_TOP)
        pygame.gfxdraw.aacircle(self.screen, rider_pos[0], rider_pos[1], self.SLED_RADIUS // 2, self.COLOR_SLED_TOP)
    
    def _draw_aim_cursor(self):
        start_pos = self.track_segments[-1][1]
        end_pos = self.aim_cursor
        
        # Draw dashed line
        self._draw_dashed_line(start_pos, end_pos, self.COLOR_AIM_LINE)

        # Draw cursor
        cursor_pos_int = (int(end_pos[0]), int(end_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 5, self.COLOR_AIM_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 5, self.COLOR_AIM_CURSOR)

    def _draw_dashed_line(self, p1, p2, color, dash_len=10):
        line_vec = p2 - p1
        dist = np.linalg.norm(line_vec)
        if dist == 0: return
        unit_vec = line_vec / dist
        
        current_pos = p1.copy()
        for _ in range(int(dist / (dash_len * 2))):
            end_segment = current_pos + unit_vec * dash_len
            pygame.draw.aaline(self.screen, color, current_pos, end_segment)
            current_pos += unit_vec * dash_len * 2

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            msg = "FINISH!" if self.reached_finish else "CRASHED"
            end_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, pos, count, color, speed_scale, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] > 0:
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'] * (p['life'] / 20.0))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, p['color'])
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for interactive testing ---
    pygame.display.set_caption("Sled Drawer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        mov_action = 0 # 0=none
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [mov_action, space_action, shift_action]
        
        # Only step if an action is taken
        if any(keys):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait a bit before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The environment's observation is already the rendered image
        # We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit interactive loop speed

    env.close()