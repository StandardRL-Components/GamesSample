
# Generated: 2025-08-27T13:56:43.069935
# Source Brief: brief_00536.md
# Brief Index: 536

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Vector Math Helper Functions ---
def v_add(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])

def v_sub(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])

def v_mul_scalar(v, s):
    return (v[0] * s, v[1] * s)

def v_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def v_mag_sq(v):
    return v[0]**2 + v[1]**2

def v_mag(v):
    mag_sq = v_mag_sq(v)
    return math.sqrt(mag_sq) if mag_sq > 0 else 0

def v_normalize(v):
    mag = v_mag(v)
    return (v[0] / mag, v[1] / mag) if mag > 0 else (0, 0)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows draw different line segments. "
        "Spacebar draws a long flat line. Shift draws a long steep downward ramp. "
        "Guide the sled to the green finish line!"
    )

    game_description = (
        "A physics-based sledding game. Draw lines to create a track for your sled, "
        "navigating a procedurally generated mountain. Balance speed and safety to "
        "reach the finish line before time runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30 # For timer calculations
        self.MAX_TIME_SECONDS = 15
        self.MAX_STEPS = 1000
        self.PHYSICS_SUBSTEPS = 15

        # --- Colors and Fonts ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_MOUNTAIN_1 = (25, 30, 45)
        self.COLOR_MOUNTAIN_2 = (35, 40, 55)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_SLED_GLOW = (200, 200, 255, 50)
        self.COLOR_LINE = (255, 50, 50)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_PARTICLE = (200, 220, 255)

        # --- Physics Parameters ---
        self.GRAVITY = 0.15
        self.SLED_RADIUS = 6
        self.LINE_FRICTION = 0.995 # Velocity multiplier on contact
        self.LINE_RESTITUTION = 0.4 # Bounciness

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_speed = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left_steps = 0
        self.sled_pos = [0.0, 0.0]
        self.sled_vel = [0.0, 0.0]
        self.lines = []
        self.particles = []
        self.last_line_end = (0, 0)
        self.finish_line_x = 0
        self.mountains = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left_steps = self.MAX_TIME_SECONDS * self.FPS

        # --- Sled State ---
        self.sled_pos = [60.0, 85.0]
        self.sled_vel = [3.0, 0.0]
        
        # --- Track State ---
        self.lines = []
        start_platform = ((20, 100), (120, 100))
        self.lines.append(start_platform)
        self.last_line_end = start_platform[1]
        
        # --- Finish Line ---
        self.finish_line_x = self.SCREEN_WIDTH - 40

        # --- Particles ---
        self.particles = []

        # --- Generate Background ---
        self.mountains = []
        for i in range(3):
            points = [(0, self.np_random.integers(150, 250))]
            for x in range(0, self.SCREEN_WIDTH + 101, 100):
                points.append((x + self.np_random.integers(-30, 30), self.np_random.integers(100, 300 - i * 50)))
            points.append((self.SCREEN_WIDTH, self.np_random.integers(150, 250)))
            points.append((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            points.append((0, self.SCREEN_HEIGHT))
            color = [c + i*10 for c in self.COLOR_BG]
            self.mountains.append({'points': points, 'color': color})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Action: Draw Line ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        line_drawn = self._handle_line_drawing(movement, space_held, shift_held)
        
        step_reward = -0.1 if line_drawn else 0.0

        # --- 2. Run Physics Simulation ---
        old_sled_x = self.sled_pos[0]
        for _ in range(self.PHYSICS_SUBSTEPS):
            if self.game_over: break
            self._apply_gravity()
            self._handle_collisions()
            self.sled_pos = v_add(self.sled_pos, self.sled_vel)
        
        # --- 3. Update Particles ---
        self._update_particles()
        
        # --- 4. Calculate Reward ---
        distance_reward = (self.sled_pos[0] - old_sled_x) * 0.1
        step_reward += distance_reward

        # --- 5. Update Game State & Check Termination ---
        self.steps += 1
        self.time_left_steps -= 1
        terminated = False
        
        # Win Condition
        if self.sled_pos[0] >= self.finish_line_x:
            terminated = True
            time_bonus = 20 + 80 * max(0, self.time_left_steps) / (self.MAX_TIME_SECONDS * self.FPS)
            step_reward += time_bonus
            # // Win sound effect
        
        # Loss Conditions
        elif self.time_left_steps <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            step_reward = -100.0
        elif not (0 < self.sled_pos[0] < self.SCREEN_WIDTH and -self.SLED_RADIUS < self.sled_pos[1] < self.SCREEN_HEIGHT):
            terminated = True
            step_reward = -100.0
            # // Crash sound effect

        self.game_over = terminated
        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_line_drawing(self, movement, space_held, shift_held):
        line_defs = {
            'shift': {'len': 100, 'angle_deg': -35}, # Long steep down
            'space': {'len': 80, 'angle_deg': 0},    # Long flat
            1: {'len': 40, 'angle_deg': 35},         # Up -> Short steep up
            2: {'len': 40, 'angle_deg': -35},        # Down -> Short steep down
            3: {'len': 40, 'angle_deg': 0},          # Left -> Short flat
            4: {'len': 60, 'angle_deg': 0},          # Right -> Medium flat
        }

        line_key = None
        if shift_held: line_key = 'shift'
        elif space_held: line_key = 'space'
        elif movement > 0: line_key = movement
        
        if line_key is not None:
            spec = line_defs[line_key]
            angle_rad = math.radians(spec['angle_deg'])
            start_point = self.last_line_end
            end_point = (
                start_point[0] + spec['len'] * math.cos(angle_rad),
                start_point[1] + spec['len'] * math.sin(angle_rad)
            )
            
            # Clamp line to screen bounds to prevent drawing out of view
            end_point = (
                max(0, min(self.SCREEN_WIDTH, end_point[0])),
                max(0, min(self.SCREEN_HEIGHT, end_point[1]))
            )

            self.lines.append((start_point, end_point))
            self.last_line_end = end_point
            
            # Keep line list from growing indefinitely
            if len(self.lines) > 20:
                self.lines.pop(0)
            
            return True # Line was drawn
        return False # No line drawn

    def _apply_gravity(self):
        self.sled_vel = v_add(self.sled_vel, (0, self.GRAVITY))

    def _handle_collisions(self):
        for p1, p2 in self.lines:
            line_vec = v_sub(p2, p1)
            line_mag_sq = v_mag_sq(line_vec)
            if line_mag_sq == 0: continue

            sled_to_p1 = v_sub(self.sled_pos, p1)
            t = v_dot(sled_to_p1, line_vec) / line_mag_sq
            t = max(0, min(1, t)) # Clamp to segment

            closest_point = v_add(p1, v_mul_scalar(line_vec, t))
            dist_vec = v_sub(self.sled_pos, closest_point)
            dist_sq = v_mag_sq(dist_vec)

            if dist_sq < self.SLED_RADIUS**2:
                # // Sled hit a line sound effect
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-6
                penetration = self.SLED_RADIUS - dist
                normal = v_normalize(dist_vec)

                # Resolve intersection
                self.sled_pos = v_add(self.sled_pos, v_mul_scalar(normal, penetration))

                # Apply impulse (reflection)
                v_dot_n = v_dot(self.sled_vel, normal)
                impulse = -(1 + self.LINE_RESTITUTION) * v_dot_n
                self.sled_vel = v_add(self.sled_vel, v_mul_scalar(normal, impulse))
                
                # Apply friction
                self.sled_vel = v_mul_scalar(self.sled_vel, self.LINE_FRICTION)

                # Create particles
                self._create_collision_particles(self.sled_pos, normal)

    def _create_collision_particles(self, pos, normal, num_particles=5):
        for _ in range(num_particles):
            angle = math.atan2(normal[1], normal[0]) + self.np_random.uniform(-0.8, 0.8)
            speed = self.np_random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for mountain in self.mountains:
            pygame.gfxdraw.aapolygon(self.screen, mountain['points'], mountain['color'])
            pygame.gfxdraw.filled_polygon(self.screen, mountain['points'], mountain['color'])

    def _render_game_elements(self):
        # Finish Line
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, y), (self.finish_line_x, y + 10), 3)

        # Player-drawn Lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, p1, p2, True)
            pygame.draw.line(self.screen, self.COLOR_LINE, p1, p2, 4)

        # Sled Glow
        glow_surf = pygame.Surface((self.SLED_RADIUS * 4, self.SLED_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_SLED_GLOW, (self.SLED_RADIUS * 2, self.SLED_RADIUS * 2), self.SLED_RADIUS * 2)
        self.screen.blit(glow_surf, (int(self.sled_pos[0] - self.SLED_RADIUS * 2), int(self.sled_pos[1] - self.SLED_RADIUS * 2)))

        # Sled
        pygame.gfxdraw.aacircle(self.screen, int(self.sled_pos[0]), int(self.sled_pos[1]), self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.filled_circle(self.screen, int(self.sled_pos[0]), int(self.sled_pos[1]), self.SLED_RADIUS, self.COLOR_SLED)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*self.COLOR_PARTICLE, alpha)
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (p['pos'][0] - size, p['pos'][1] - size))


    def _render_ui(self):
        # Distance
        dist_text = self.font_ui.render(f"DISTANCE: {int(self.sled_pos[0])}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 10))

        # Time
        time_str = f"TIME: {max(0, self.time_left_steps / self.FPS):.1f}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Speed
        speed = v_mag(self.sled_vel) * 5 # Scale for better display value
        speed_text = self.font_speed.render(f"{int(speed)} km/h", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH // 2 - speed_text.get_width() // 2, self.SCREEN_HEIGHT - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": int(self.sled_pos[0]),
            "time_left": round(max(0, self.time_left_steps / self.FPS), 2)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Sledder")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("--- Resetting Game ---")
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            # For human play, we step on key presses, not every frame
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # We always need to get the latest observation to render
        # In auto_advance=False, this just re-renders the current state
        frame = env._get_observation()
        
        # Convert from (H, W, C) to (W, H, C) and draw
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()