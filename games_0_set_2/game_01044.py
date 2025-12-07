import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string:
    user_guide = (
        "Controls: Arrows move the drawing cursor. Hold Space to draw lines. "
        "Hold Shift to snap the cursor to the sled."
    )

    # Short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Draw tracks for a sled to ride on, guiding it from the start "
        "to the finish line before time runs out. Master physics to build the fastest, most stable tracks."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60  # Simulation FPS, not render FPS
    TIME_LIMIT_SECONDS = 30
    MAX_EPISODE_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 22)
    COLOR_GRID = (30, 35, 40)
    COLOR_START = (0, 255, 127) # Spring Green
    COLOR_FINISH = (255, 69, 58) # Bright Red
    COLOR_SLED = (0, 122, 255) # Bright Blue
    COLOR_TRACK = (255, 255, 255)
    COLOR_CURSOR = (255, 204, 0) # Gold
    COLOR_SPARK = (255, 214, 10)
    COLOR_TRAIL = (0, 122, 255, 50) # Translucent sled color
    COLOR_UI_BG = (0, 0, 0, 100)
    COLOR_UI_TEXT = (255, 255, 255)

    # Physics
    GRAVITY = 0.15
    FRICTION = 0.995 # Multiplier for velocity on contact
    BOUNCE = 0.4 # Energy retained on collision
    SLED_RADIUS = 6
    CURSOR_SPEED = 8
    PHYSICS_SUBSTEPS = 8 # Number of physics updates per agent step

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
        self.font_ui = pygame.font.SysFont("sans-serif", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)

        self.difficulty_level = 0.0
        self.total_wins = 0
        
        # This will be properly seeded in reset()
        self.np_random = None

        # Initialize state variables to be defined in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = 0
        self.sled_pos = np.zeros(2, dtype=float)
        self.sled_vel = np.zeros(2, dtype=float)
        self.pen_pos = np.zeros(2, dtype=float)
        self.last_pen_pos = np.zeros(2, dtype=float)
        self.is_drawing = False
        self.terrain_lines = []
        self.drawn_lines = []
        self.particles = []
        self.trails = []
        self.finish_line_x = 0
        self.checkpoint_x = 0
        self.checkpoint_reached = False

        # The initial reset() call will seed the RNG
        # self.reset() is called by the environment wrapper, no need to call it here.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Procedural Terrain Generation ---
        self.terrain_lines = []
        start_y = self.np_random.uniform(100, self.SCREEN_HEIGHT - 100)
        start_pos = np.array([50.0, start_y])
        self.sled_pos = start_pos.copy()
        self.sled_vel = np.array([1.0, 0.0]) # Give it a small initial push

        # Place finish line far right
        self.finish_line_x = self.SCREEN_WIDTH - 50
        finish_y = self.np_random.uniform(100, self.SCREEN_HEIGHT - 100)
        
        # Checkpoint
        self.checkpoint_x = (start_pos[0] + self.finish_line_x) / 2
        self.checkpoint_reached = False

        # Generate some simple hills/gaps based on difficulty
        num_features = 2 + int(self.difficulty_level / 2)
        feature_points = [start_pos]
        for i in range(1, num_features):
            px = start_pos[0] + i * (self.finish_line_x - start_pos[0]) / num_features
            py = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            # Make gaps more likely with higher difficulty
            if self.np_random.random() < self.difficulty_level * 0.1:
                 # Create a gap
                 prev_point = feature_points[-1]
                 gap_start = prev_point.copy()
                 gap_start[0] += 30
                 self.terrain_lines.append((prev_point, gap_start))
                 feature_points.append(np.array([px,py]))
            else:
                feature_points.append(np.array([px,py]))
        feature_points.append(np.array([self.finish_line_x, finish_y]))

        for i in range(len(feature_points) - 1):
             # Don't connect over gaps that were already made
            if len(self.terrain_lines) == 0 or not np.array_equal(self.terrain_lines[-1][1], feature_points[i]):
                 self.terrain_lines.append((feature_points[i], feature_points[i+1]))

        # --- Reset other states ---
        self.pen_pos = self.sled_pos.copy() + np.array([20, -20])
        self.last_pen_pos = self.pen_pos.copy()
        self.is_drawing = False
        self.drawn_lines = []
        self.particles = []
        self.trails = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action
        space_held = space_action == 1
        shift_held = shift_action == 1

        # --- 1. Handle Agent Action ---
        # Update pen position based on arrow keys
        if movement == 1: self.pen_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.pen_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.pen_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.pen_pos[0] += self.CURSOR_SPEED  # Right
        self.pen_pos[0] = np.clip(self.pen_pos[0], 0, self.SCREEN_WIDTH)
        self.pen_pos[1] = np.clip(self.pen_pos[1], 0, self.SCREEN_HEIGHT)
        
        # Snap pen to sled
        if shift_held:
            self.pen_pos = self.sled_pos.copy()

        # Draw line segment if space is held
        if space_held:
            # Start a new line segment if we weren't drawing before
            if not self.is_drawing:
                self.last_pen_pos = self.pen_pos.copy()
                self.is_drawing = True
            
            # Add the new segment if it has some length
            if np.linalg.norm(self.pen_pos - self.last_pen_pos) > 1:
                self.drawn_lines.append((self.last_pen_pos.copy(), self.pen_pos.copy()))
                self.last_pen_pos = self.pen_pos.copy()
        else:
            self.is_drawing = False

        # --- 2. Simulate Physics ---
        initial_sled_x = self.sled_pos[0]
        
        for _ in range(self.PHYSICS_SUBSTEPS):
            if self.game_over: break
            self._update_physics()
        
        # --- 3. Calculate Reward ---
        reward = 0
        
        # Reward for forward progress
        progress = self.sled_pos[0] - initial_sled_x
        reward += progress * 0.1

        # Penalty for being slow (using speed from last physics step)
        speed = np.linalg.norm(self.sled_vel)
        if speed < 1.0:
            reward -= 0.1 # Scaled from brief's -1

        # Small time penalty
        reward -= 0.01

        # --- 4. Update Game State & Check Termination ---
        self.steps += 1
        self.time_left -= self.PHYSICS_SUBSTEPS
        
        terminated = False
        
        # Checkpoint reward
        if not self.checkpoint_reached and self.sled_pos[0] > self.checkpoint_x:
            reward += 5
            self.checkpoint_reached = True
            # sfx: checkpoint sound
        
        # Win condition
        if self.sled_pos[0] >= self.finish_line_x:
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
            self.total_wins += 1
            self.difficulty_level = min(10.0, self.difficulty_level + 0.2)
            # sfx: win fanfare
        
        # Crash condition
        if not (0 <= self.sled_pos[0] <= self.SCREEN_WIDTH and -self.SLED_RADIUS <= self.sled_pos[1] <= self.SCREEN_HEIGHT + self.SLED_RADIUS):
            reward -= 10
            terminated = True
            self.game_over = True
            # sfx: crash sound
            
        # Time's up or max steps
        if self.time_left <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            if not terminated: # Don't double-penalize
                reward -= 10
            terminated = True
            self.game_over = True
            # sfx: timeout buzzer

        self.score += reward

        truncated = False # This environment does not truncate based on time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        # Update position
        self.sled_pos += self.sled_vel

        # Collision detection and response
        all_lines = self.terrain_lines + self.drawn_lines
        for p1, p2 in all_lines:
            p1 = np.array(p1)
            p2 = np.array(p2)
            
            # Find closest point on line segment to sled
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = max(0, min(1, np.dot(self.sled_pos - p1, line_vec) / line_len_sq))
            closest_point = p1 + t * line_vec
            
            dist_vec = self.sled_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.SLED_RADIUS ** 2:
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
                
                # Resolve penetration
                penetration = self.SLED_RADIUS - dist
                normal = dist_vec / dist if dist > 0 else np.array([0.0, -1.0])
                self.sled_pos += normal * penetration

                # Reflect velocity
                velocity_component = np.dot(self.sled_vel, normal)
                self.sled_vel -= (1 + self.BOUNCE) * velocity_component * normal
                
                # Apply friction
                self.sled_vel *= self.FRICTION

                # Create sparks
                if np.linalg.norm(self.sled_vel) > 1.5:
                    # sfx: grind/spark sound
                    for _ in range(3):
                        spark_vel = self.sled_vel * 0.2 + self.np_random.standard_normal(2) * 1.5
                        self.particles.append({
                            "pos": self.sled_pos.copy(),
                            "vel": spark_vel,
                            "life": 20,
                            "color": self.COLOR_SPARK
                        })
                break # Handle one collision per substep
        
        # Add trail particle
        if len(self.trails) == 0 or np.linalg.norm(self.sled_pos - self.trails[-1]["pos"]) > 3:
            self.trails.append({"pos": self.sled_pos.copy(), "life": 30})

    def _update_particles_and_trails(self):
        # Update sparks
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

        # Update trails
        active_trails = []
        for t in self.trails:
            t["life"] -= 1
            if t["life"] > 0:
                active_trails.append(t)
        self.trails = active_trails

    def _get_observation(self):
        # --- 1. Clear screen and draw background ---
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._update_particles_and_trails()

        # --- 2. Render game elements ---
        self._render_game()

        # --- 3. Render UI overlay ---
        self._render_ui()

        # --- 4. Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Trails
        for i in range(len(self.trails) - 1):
            p1 = self.trails[i]
            p2 = self.trails[i+1]
            alpha = int(255 * (p1['life'] / 30))
            # FIX: Use pygame.draw.aaline which exists, unlike pygame.gfxdraw.aaline
            # It also requires points as tuples/lists, not separate coordinates.
            start_pos = (p1['pos'][0], p1['pos'][1])
            end_pos = (p2['pos'][0], p2['pos'][1])
            color = (*self.COLOR_SLED, alpha)
            pygame.draw.aaline(self.screen, color, start_pos, end_pos)

        # Terrain lines
        for p1, p2 in self.terrain_lines:
            is_start = p1[0] <= 50
            is_finish = p2[0] >= self.finish_line_x
            color = self.COLOR_START if is_start else (self.COLOR_FINISH if is_finish else self.COLOR_TRACK)
            pygame.draw.aaline(self.screen, color, p1, p2, 2)
        
        # Finish line post
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.SCREEN_HEIGHT), 2)
        
        # Checkpoint line
        if not self.checkpoint_reached:
             pygame.draw.line(self.screen, (255,255,255,50), (self.checkpoint_x, 0), (self.checkpoint_x, self.SCREEN_HEIGHT), 1)

        # Drawn lines
        for p1, p2 in self.drawn_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)

        # Sled
        sled_x, sled_y = int(self.sled_pos[0]), int(self.sled_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            # Use a surface with SRCALPHA for proper alpha blending
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, color)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - 2, int(p['pos'][1]) - 2), special_flags=pygame.BLEND_RGBA_ADD)


        # Drawing Cursor
        if not self.game_over:
            cx, cy = int(self.pen_pos[0]), int(self.pen_pos[1])
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 5, cy), (cx + 5, cy), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 5), (cx, cy + 5), 1)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 30), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Time
        time_str = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=5)
        self.screen.blit(time_text, time_rect)

        # Speed
        speed = np.linalg.norm(self.sled_vel)
        speed_text = self.font_ui.render(f"SPEED: {speed:.1f}", True, self.COLOR_UI_TEXT)
        speed_rect = speed_text.get_rect(right=self.SCREEN_WIDTH - 10, y=5)
        self.screen.blit(speed_text, speed_rect)
        
        # Game Over Message
        if self.game_over:
            msg = "FINISH!" if self.win else "CRASHED!"
            color = self.COLOR_START if self.win else self.COLOR_FINISH
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "wins": self.total_wins,
            "difficulty": self.difficulty_level
        }

    def close(self):
        pygame.quit()


# Example usage for interactive testing:
if __name__ == '__main__':
    # This block will not run in the sandboxed evaluation environment.
    # It is for local testing.
    
    # Un-set the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    pygame.display.set_caption("Line Rider Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    running = True
    game_over_display = False
    
    print("\n" + "="*50)
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset the environment.")
    print("="*50 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                game_over_display = False
        
        if not game_over_display:
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Space
            if keys[pygame.K_SPACE]: action[1] = 1
            
            # Shift
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Episode finished! Score: {info['score']:.2f}, Steps: {info['steps']}")
                game_over_display = True
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit interactive play to 30 FPS

    env.close()