
# Generated: 2025-08-27T14:04:30.508967
# Source Brief: brief_00572.md
# Brief Index: 572

        
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
        "Use arrow keys to draw track segments. Hold Space to draw a longer segment. Hold Shift to undo the last segment."
    )

    game_description = (
        "Draw a path for your sledder to ride across the landscape. Reach the finish line before time runs out, but be careful - if the rider leaves your track, they'll crash!"
    )

    auto_advance = False
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WORLD_WIDTH = WIDTH * 3
    NUM_STAGES = 3
    MAX_TIME = 100.0
    MAX_STEPS = 2000
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (210, 235, 255) # Lighter Sky
    COLOR_TERRAIN = (139, 69, 19)    # Saddle Brown
    COLOR_TRACK = (20, 20, 20)       # Near Black
    COLOR_RIDER = (255, 0, 0)        # Red
    COLOR_CHECKPOINT = (0, 128, 0)   # Green
    COLOR_CHECKPOINT_REACHED = (255, 215, 0) # Gold
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)
    COLOR_PARTICLE = (160, 82, 45) # Sienna

    # Physics & Gameplay
    GRAVITY = 0.3
    RIDER_RADIUS = 8
    FRICTION = 0.995
    SIM_TIME_PER_STEP = 0.5  # Seconds of game time per step() call
    SIM_SUBSTEPS = 15
    DRAW_LENGTH = 30
    DRAW_LENGTH_BOOST = 60
    FINISH_X = WORLD_WIDTH - 50

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
        
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 48)
        
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_angle = 0
        self.track_points = []
        self.terrain_points = []
        self.checkpoints = []
        self.particles = []

        self.reset()

        # self.validate_implementation() # Uncomment to run validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.MAX_TIME
        
        self._generate_terrain()
        
        initial_y = self._get_terrain_y(50) - self.RIDER_RADIUS - 1
        self.rider_pos = pygame.Vector2(50, initial_y)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_on_track = False
        
        self.track_points = [pygame.Vector2(self.rider_pos.x, self.rider_pos.y + self.RIDER_RADIUS)]
        
        self.checkpoints = []
        for i in range(self.NUM_STAGES + 1):
            x = i * self.WIDTH
            if x == 0: x = 50 # Start line
            if x >= self.FINISH_X: x = self.FINISH_X
            y = self._get_terrain_y(x)
            self.checkpoints.append({"pos": pygame.Vector2(x, y), "reached": False})
        self.checkpoints[0]["reached"] = True # Starting checkpoint
            
        self.camera_x = 0
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # 1. Handle player action (drawing/undoing track)
        if shift_held:
            if len(self.track_points) > 1:
                self.track_points.pop()
                reward -= 0.1 # Penalty for undoing
        elif movement > 0:
            reward -= 0.01 # Cost for drawing
            last_point = self.track_points[-1]
            direction_map = {
                1: pygame.Vector2(0, -1), # Up
                2: pygame.Vector2(0, 1),  # Down
                3: pygame.Vector2(-1, 0), # Left
                4: pygame.Vector2(1, 0),  # Right
            }
            direction = direction_map[movement]
            length = self.DRAW_LENGTH_BOOST if space_held else self.DRAW_LENGTH
            new_point = last_point + direction * length
            
            # Clamp to world boundaries
            new_point.x = max(0, min(self.WORLD_WIDTH, new_point.x))
            new_point.y = max(0, min(self.HEIGHT, new_point.y))
            
            self.track_points.append(new_point)

        # 2. Run physics simulation
        sim_reward, terminated_by_sim = self._run_simulation()
        reward += sim_reward
        
        # 3. Update game state
        self.steps += 1
        self.time_left -= self.SIM_TIME_PER_STEP
        
        # 4. Check for termination conditions
        terminated = terminated_by_sim or self.time_left <= 0 or self.steps >= self.MAX_STEPS or self.win
        if terminated and not self.game_over:
            self.game_over = True
            if terminated_by_sim and not self.win: # Crash
                reward -= 50
                self._create_crash_particles()
                # sfx: crash_sound
            elif self.win:
                reward += 100 # Final win bonus
            elif self.time_left <= 0:
                reward -= 10 # Time out penalty

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _run_simulation(self):
        reward = 0
        crashed = False
        dt = self.SIM_TIME_PER_STEP / self.SIM_SUBSTEPS
        old_x = self.rider_pos.x

        for _ in range(self.SIM_SUBSTEPS):
            if crashed: break

            # Apply gravity
            self.rider_vel.y += self.GRAVITY
            
            # Find surface (track or terrain)
            track_y, track_normal = self._get_track_surface_at(self.rider_pos.x)
            terrain_y = self._get_terrain_y(self.rider_pos.x)
            
            on_track = track_y is not None and self.rider_pos.y + self.RIDER_RADIUS >= track_y - 2
            
            if on_track:
                ground_y = track_y
                ground_normal = track_normal
                self.rider_on_track = True
            else:
                ground_y = terrain_y
                # Approximate terrain normal (less critical)
                p1 = pygame.Vector2(self.rider_pos.x - 1, self._get_terrain_y(self.rider_pos.x - 1))
                p2 = pygame.Vector2(self.rider_pos.x + 1, self._get_terrain_y(self.rider_pos.x + 1))
                ground_normal = (p2 - p1).normalize().rotate(90)
                self.rider_on_track = False

            # Collision and response
            if self.rider_pos.y + self.RIDER_RADIUS > ground_y:
                if not self.rider_on_track:
                    crashed = True
                    break

                # On track: collision response
                self.rider_pos.y = ground_y - self.RIDER_RADIUS
                
                # Project velocity along the track tangent
                tangent = ground_normal.rotate(-90)
                speed = self.rider_vel.dot(tangent)
                self.rider_vel = tangent * speed

                # Apply friction
                self.rider_vel *= self.FRICTION
                
                self.rider_angle = math.degrees(math.atan2(tangent.y, tangent.x))

            else: # Airborne
                self.rider_on_track = False
                self.rider_angle = math.degrees(math.atan2(self.rider_vel.y, self.rider_vel.x))
            
            # Update position
            self.rider_pos += self.rider_vel * dt * 5 # dt is small, scale up for movement
        
        # Post-simulation checks
        # Reward for forward movement
        reward += (self.rider_pos.x - old_x) * 0.1

        # Checkpoint rewards
        for cp in self.checkpoints:
            if not cp["reached"] and self.rider_pos.x >= cp["pos"].x:
                cp["reached"] = True
                reward += 50 if cp["pos"].x >= self.FINISH_X else 10
                # sfx: checkpoint_reached_sound
                if cp["pos"].x >= self.FINISH_X:
                    self.win = True

        # Clamp rider to world
        if self.rider_pos.x < 0 or self.rider_pos.x > self.WORLD_WIDTH:
            crashed = True
            
        return reward, crashed

    def _get_observation(self):
        # Update camera to follow player, with some smoothing
        target_cam_x = self.rider_pos.x - self.WIDTH / 3
        self.camera_x += (target_cam_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.WORLD_WIDTH - self.WIDTH, self.camera_x))

        # --- Render background ---
        self._render_background()

        # --- Render game elements (with camera offset) ---
        self._render_terrain()
        self._render_track()
        self._render_checkpoints()
        self._render_particles()
        self._render_rider()
        
        # --- Render UI (fixed position) ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            mix = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - mix) + self.COLOR_BG_BOTTOM[0] * mix,
                self.COLOR_BG_TOP[1] * (1 - mix) + self.COLOR_BG_BOTTOM[1] * mix,
                self.COLOR_BG_TOP[2] * (1 - mix) + self.COLOR_BG_BOTTOM[2] * mix,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
    
    def _render_terrain(self):
        cam_x = int(self.camera_x)
        
        # Find which points are on screen
        start_idx = max(0, cam_x - 1)
        end_idx = min(len(self.terrain_points), cam_x + self.WIDTH + 2)
        
        if end_idx > start_idx + 1:
            on_screen_points = [(p[0] - cam_x, p[1]) for p in self.terrain_points[start_idx:end_idx]]
            pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, on_screen_points + [(on_screen_points[-1][0], self.HEIGHT), (on_screen_points[0][0], self.HEIGHT)])

    def _render_track(self):
        if len(self.track_points) > 1:
            cam_x = self.camera_x
            on_screen_points = []
            for p in self.track_points:
                if cam_x - 200 < p.x < cam_x + self.WIDTH + 200: # Culling with buffer
                    on_screen_points.append((p.x - cam_x, p.y))
            
            if len(on_screen_points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, on_screen_points, 3)

    def _render_checkpoints(self):
        for cp in self.checkpoints:
            x, y = cp["pos"].x - self.camera_x, cp["pos"].y
            if 0 < x < self.WIDTH:
                color = self.COLOR_CHECKPOINT_REACHED if cp["reached"] else self.COLOR_CHECKPOINT
                pygame.draw.line(self.screen, color, (x, y), (x, y - 40), 3)
                pygame.draw.polygon(self.screen, color, [(x, y - 40), (x + 15, y - 35), (x, y - 30)])

    def _render_rider(self):
        x, y = int(self.rider_pos.x - self.camera_x), int(self.rider_pos.y)
        
        # Rider body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # Rider direction indicator (rotated line)
        rad_angle = math.radians(self.rider_angle)
        end_x = x + math.cos(rad_angle) * (self.RIDER_RADIUS + 2)
        end_y = y + math.sin(rad_angle) * (self.RIDER_RADIUS + 2)
        pygame.draw.aaline(self.screen, (255, 255, 255), (x, y), (end_x, end_y))

    def _render_particles(self):
        cam_x = self.camera_x
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
                px, py = int(p['pos'].x - cam_x), int(p['pos'].y)
                pygame.draw.rect(self.screen, p['color'], (px, py, size, size))
    
    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score and Time
        draw_text(f"Score: {int(self.score)}", self.font_ui, self.COLOR_UI_TEXT, (10, 10), self.COLOR_UI_SHADOW)
        time_str = f"Time: {max(0, self.time_left):.1f}"
        time_size = self.font_ui.size(time_str)
        draw_text(time_str, self.font_ui, self.COLOR_UI_TEXT, (self.WIDTH - time_size[0] - 10, 10), self.COLOR_UI_SHADOW)

        # Checkpoint indicators
        num_cp = len(self.checkpoints)
        total_width = num_cp * 20
        start_x = (self.WIDTH - total_width) / 2
        for i, cp in enumerate(self.checkpoints):
            x = int(start_x + i * 20)
            y = self.HEIGHT - 20
            color = self.COLOR_CHECKPOINT_REACHED if cp["reached"] else self.COLOR_TRACK
            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, self.COLOR_UI_TEXT)

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "CRASHED!"
            if self.time_left <= 0 and not self.win: msg = "TIME'S UP!"
            draw_text(msg, self.font_msg, self.COLOR_UI_TEXT, (self.WIDTH/2 - self.font_msg.size(msg)[0]/2, self.HEIGHT/2 - 50), self.COLOR_UI_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "rider_x": self.rider_pos.x,
        }

    def _generate_terrain(self):
        self.terrain_points = []
        
        # Use the numpy random generator for reproducibility
        octaves = [
            (self.np_random.uniform(0.002, 0.005), self.np_random.uniform(50, 80)),
            (self.np_random.uniform(0.01, 0.015), self.np_random.uniform(15, 25)),
            (self.np_random.uniform(0.05, 0.08), self.np_random.uniform(3, 8)),
        ]
        
        for x in range(self.WORLD_WIDTH + 1):
            y_offset = 0
            stage_progress = x / self.WIDTH
            
            # Increase amplitude based on stage
            amplitude_multiplier = 1.0 + min(2.0, stage_progress) * 0.5 

            for i, (freq, amp) in enumerate(octaves):
                phase = i * self.np_random.uniform(5, 15)
                y_offset += math.sin(x * freq + phase) * amp * amplitude_multiplier
            
            final_y = self.HEIGHT * 0.7 + y_offset
            self.terrain_points.append((x, final_y))

    def _get_terrain_y(self, x):
        x = max(0, min(self.WORLD_WIDTH - 1, x))
        x_idx = int(x)
        p1 = self.terrain_points[x_idx]
        p2 = self.terrain_points[x_idx + 1]
        # Linear interpolation
        return p1[1] + (p2[1] - p1[1]) * (x - x_idx)

    def _get_track_surface_at(self, x):
        if len(self.track_points) < 2:
            return None, None

        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            if (p1.x <= x < p2.x) or (p2.x <= x < p1.x):
                # Interpolate y-position on the segment
                if abs(p2.x - p1.x) < 1e-6: # Vertical line
                    y = p1.y if p1.y < p2.y else p2.y
                else:
                    t = (x - p1.x) / (p2.x - p1.x)
                    y = p1.y + t * (p2.y - p1.y)
                
                normal = (p2 - p1).normalize().rotate(90)
                if normal.y > 0:
                    normal *= -1 # Normal should always point upwards
                return y, normal
        
        return None, None
    
    def _create_crash_particles(self):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': self.rider_pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 40),
                'max_life': 40,
                'size': self.np_random.integers(3, 7),
                'color': self.COLOR_PARTICLE
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Sled Rider")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0
    
    print(GameEnv.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Keydown events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r: # Reset
                    env.reset()
                    terminated = False
            
            # Keyup events
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                elif event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        if env.game_over:
            # Wait for reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
                    terminated = False
        else:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment returns an RGB array, we need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        # The obs is (height, width, 3) so we just need to transpose it for pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate here
        clock.tick(10) # Run at 10 actions per second for human play
        
    env.close()