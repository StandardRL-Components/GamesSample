
# Generated: 2025-08-28T05:58:09.816153
# Source Brief: brief_05746.md
# Brief Index: 5746

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to set line direction. Space to draw a longer line, Shift for a shorter one. The goal is to draw a track to guide the sled to the green finish line."
    )

    game_description = (
        "A physics-based puzzle game inspired by Line Rider. Draw track segments for a sled to ride on, navigating from the start to the finish. Master gravity and momentum to create the perfect path."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals and Theming
        self.FONT_UI = pygame.font.SysFont("monospace", 18, bold=True)
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 48, 56)
        self.COLOR_TRACK = (50, 150, 255)
        self.COLOR_SLED = (255, 50, 100)
        self.COLOR_FINISH = (50, 255, 150)
        self.COLOR_OBSTACLE = (100, 100, 110)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255, 100)

        # Game constants
        self.GRAVITY = pygame.Vector2(0, 0.15)
        self.MAX_SIM_TIME = 20.0  # seconds
        self.MAX_STEPS = 500
        self.MAX_SIM_ITERATIONS = 600 # Max physics updates per step()
        
        # Initialize state variables
        self.sled = None
        self.lines = None
        self.obstacles = None
        self.particles = None
        self.drawing_cursor = None
        self.start_pos = None
        self.finish_rect = None
        self.checkpoint_pos = None
        
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.sim_time = 0.0
        self.game_over = False
        self.checkpoint_reached = False
        
        # Sled state
        self.start_pos = pygame.Vector2(80, 80)
        self.sled = {
            "pos": pygame.Vector2(self.start_pos),
            "vel": pygame.Vector2(0, 0),
            "radius": 8,
            "on_ground": False
        }
        
        # World elements
        self.lines = []
        self.obstacles = [pygame.Rect(300, 250, 40, 150)]
        self.particles = []
        self.drawing_cursor = pygame.Vector2(self.start_pos)
        self.finish_rect = pygame.Rect(self.WIDTH - 100, self.HEIGHT - 80, 20, 80)
        self.checkpoint_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 100)
        
        # Initial state for rewards
        self.max_x_reached = self.sled["pos"].x

        # Create a simple starting ramp
        self.lines.append((pygame.Vector2(40, 100), pygame.Vector2(120, 90)))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Action Handling: Draw a new line ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        base_len = 25
        if space_held:
            line_len = base_len * 2.0
        elif shift_held:
            line_len = base_len * 0.5
        else:
            line_len = base_len

        direction = pygame.Vector2(0, 0)
        if movement == 1: direction.y = -1  # Up
        elif movement == 2: direction.y = 1   # Down
        elif movement == 3: direction.x = -1  # Left
        elif movement == 4: direction.x = 1   # Right
        
        start_point = pygame.Vector2(self.drawing_cursor)
        if direction.length() > 0:
            end_point = start_point + direction.normalize() * line_len
        else: # No-op
            end_point = start_point + pygame.Vector2(5, 0) # Draw tiny horizontal line
        
        # Clamp line to screen bounds
        end_point.x = max(0, min(self.WIDTH, end_point.x))
        end_point.y = max(0, min(self.HEIGHT, end_point.y))

        self.lines.append((start_point, end_point))
        self.drawing_cursor.update(end_point)

        self.steps += 1
        reward = -0.01  # Cost for drawing a line

        # --- 2. Physics Simulation ---
        terminated = False
        termination_reward = 0
        settled_counter = 0
        
        for i in range(self.MAX_SIM_ITERATIONS):
            if terminated: break
            
            self._update_sled_physics()
            self._update_particles()
            
            self.sim_time += 1.0 / 60.0 # Assume 60fps physics sim

            # Check for termination conditions during simulation
            if not self.screen.get_rect().collidepoint(self.sled["pos"]):
                terminated = True
                termination_reward = -100 # Crashed (out of bounds)
                # Sound: crash_sound.play()
                break
            if self.finish_rect.collidepoint(self.sled["pos"]):
                terminated = True
                termination_reward = 100 # Won
                # Sound: win_sound.play()
                break
            for obs in self.obstacles:
                if obs.collidepoint(self.sled["pos"]):
                    terminated = True
                    termination_reward = -100 # Crashed (obstacle)
                    # Sound: crash_sound.play()
                    break
            if self.sim_time >= self.MAX_SIM_TIME:
                terminated = True
                termination_reward = -50 # Timed out
                break

            # Check if sled has settled to end simulation for this step
            if self.sled["vel"].length_squared() < 0.01 and self.sled["on_ground"]:
                settled_counter += 1
            else:
                settled_counter = 0
            
            if settled_counter > 15: # Settled for 1/4 second
                break
        
        reward += termination_reward

        # --- 3. Post-Simulation Rewards ---
        if not terminated:
            # Reward for horizontal progress
            progress = self.sled["pos"].x - self.max_x_reached
            if progress > 0:
                reward += progress * 0.1
                self.max_x_reached = self.sled["pos"].x

            # Reward for reaching checkpoint
            if not self.checkpoint_reached and self.sled["pos"].x > self.checkpoint_pos.x:
                self.checkpoint_reached = True
                reward += 5
                # Sound: checkpoint_sound.play()

        # --- 4. Finalization ---
        self.game_over = terminated or (self.steps >= self.MAX_STEPS)
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True # Episode ends due to step limit

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_sled_physics(self):
        sled = self.sled
        sled["on_ground"] = False

        # Apply gravity
        sled["vel"] += self.GRAVITY

        # Move sled
        sled["pos"] += sled["vel"]

        # Collision detection and response
        closest_dist_sq = float('inf')
        closest_line = None
        closest_point_on_line = None

        for p1, p2 in self.lines:
            dist_sq, pt = self._point_segment_dist_sq(sled["pos"], p1, p2)
            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_line = (p1, p2)
                closest_point_on_line = pt
        
        if closest_line and closest_dist_sq < sled["radius"]**2:
            sled["on_ground"] = True
            dist = math.sqrt(closest_dist_sq)
            
            # Collision response
            penetration = sled["radius"] - dist
            if dist > 0:
                normal = (sled["pos"] - closest_point_on_line).normalize()
            else: # Sled center is on the line
                line_vec = (closest_line[1] - closest_line[0]).normalize()
                normal = pygame.Vector2(-line_vec.y, line_vec.x)

            sled["pos"] += normal * penetration
            
            # Velocity response
            restitution = 0.2 # Bounciness
            friction = 0.1 # Sliding friction
            
            vn = sled["vel"].dot(normal)
            if vn < 0:
                # Reflect velocity
                sled["vel"] -= (1 + restitution) * vn * normal
                
                # Apply friction
                tangent = pygame.Vector2(normal.y, -normal.x)
                vt = sled["vel"].dot(tangent)
                sled["vel"] -= vt * friction * tangent
            
            # Spawn particles if moving on ground
            if sled["vel"].length() > 1.0:
                self._spawn_particles(sled["pos"], sled["vel"])
                # Sound: sled_sliding_loop.play()

    def _spawn_particles(self, pos, vel):
        for _ in range(2):
            p_vel = -vel.normalize() * random.uniform(0.5, 1.5) + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
            p_pos = pos + p_vel * self.sled["radius"]
            self.particles.append({
                "pos": p_pos,
                "vel": p_vel,
                "life": random.randint(15, 30),
                "size": random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _point_segment_dist_sq(self, p, a, b):
        l2 = (a - b).length_squared()
        if l2 == 0.0:
            return (p - a).length_squared(), a
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return (p - projection).length_squared(), projection

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

        # --- Game Elements ---
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_rect)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
        
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)
        
        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = self.COLOR_SLED + (alpha,)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(p["size"], p["size"]), special_flags=pygame.BLEND_RGBA_ADD)

        # --- Sled ---
        sled_pos = (int(self.sled["pos"].x), int(self.sled["pos"].y))
        pygame.gfxdraw.filled_circle(self.screen, sled_pos[0], sled_pos[1], self.sled["radius"], self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, sled_pos[0], sled_pos[1], self.sled["radius"], self.COLOR_SLED)

        # --- Drawing Cursor ---
        if not self.game_over:
            cursor_pos = (int(self.drawing_cursor.x), int(self.drawing_cursor.y))
            s = 8
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos[0] - s, cursor_pos[1]), (cursor_pos[0] + s, cursor_pos[1]), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos[0], cursor_pos[1] - s), (cursor_pos[0], cursor_pos[1] + s), 1)

        # --- UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        time_text = self.FONT_UI.render(f"TIME: {self.sim_time:.2f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))
        
        score_text = self.FONT_UI.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        steps_text = self.FONT_UI.render(f"LINES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sim_time": self.sim_time,
            "sled_pos": (self.sled["pos"].x, self.sled["pos"].y),
            "checkpoint_reached": self.checkpoint_reached,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                    action_taken = True
                if event.key == pygame.K_r: # Reset on 'r'
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                    action_taken = False # Don't step on reset
        
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Episode finished. Press 'R' to reset.")

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()