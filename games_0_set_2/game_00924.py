
# Generated: 2025-08-27T15:12:48.176552
# Source Brief: brief_00924.md
# Brief Index: 924

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to steer. Hold SPACE to accelerate and SHIFT to brake. "
        "Stay on the track and reach the finish line before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, minimalist line racer. Navigate three procedurally generated neon tracks "
        "against the clock, mastering high-speed turns and avoiding the walls to achieve a high score."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and FPS
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30 # A balance between smoothness and performance for RL

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (255, 0, 80)
        self.COLOR_PLAYER_GLOW = (255, 100, 150)
        self.COLOR_TRACK = (220, 220, 255)
        self.COLOR_TRACK_GLOW = (150, 150, 255)
        self.COLOR_FINISH = (0, 255, 120)
        self.COLOR_SPARKS = (255, 220, 0)
        self.COLOR_TRAIL = (255, 100, 0)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_SPEED_LOW = (0, 255, 0)
        self.COLOR_SPEED_HIGH = (255, 0, 0)

        # Game constants
        self.TOTAL_TRACKS = 3
        self.TRACK_TIME_LIMIT = 60.0
        self.TRACK_LENGTH = 4000
        self.TRACK_WIDTH = 100
        self.TRACK_POINTS = 200 # Number of segments for the full track

        # Player physics
        self.STEER_FORCE = 0.6
        self.Y_DRAG = 0.85
        self.MAX_Y_VEL = 10
        self.ACCELERATION = 0.2
        self.BRAKE_FORCE = 0.4
        self.DRAG = 0.99
        self.MIN_SPEED = 1.0
        self.MAX_SPEED = 25.0
        self.NEAR_MISS_THRESHOLD = 15

        # Initialize state variables
        self.player_y = 0
        self.player_y_vel = 0
        self.player_speed = 0
        self.track_progress = 0
        self.camera_x = 0
        self.current_track = 0
        self.time_left = 0
        self.track_centerline = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        
        self.reset()
        self.validate_implementation()

    def _setup_track(self):
        """Initializes state for a new track."""
        self.player_y = self.SCREEN_HEIGHT / 2
        self.player_y_vel = 0
        self.player_speed = self.MIN_SPEED
        self.track_progress = 0
        self.camera_x = 0
        self.time_left = self.TRACK_TIME_LIMIT
        self.particles.clear()
        self._generate_track()

    def _generate_track(self):
        """Generates a procedural track using a smoothed random walk."""
        self.track_centerline = []
        y = self.SCREEN_HEIGHT / 2
        curvature_difficulty = 1.0 + (self.current_track - 1) * 0.25
        dy = 0
        segment_length = self.TRACK_LENGTH / self.TRACK_POINTS

        for i in range(self.TRACK_POINTS + 1):
            target_dy = random.uniform(-2.5, 2.5) * curvature_difficulty
            dy += (target_dy - dy) * 0.1 # Smooth the change in direction
            
            y += dy
            # Prevent track from going too far off-screen
            y = np.clip(y, self.TRACK_WIDTH, self.SCREEN_HEIGHT - self.TRACK_WIDTH)

            self.track_centerline.append((i * segment_length, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.current_track = 1
        
        self._setup_track()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Handle player input and physics ---
        steer_input = 0
        if movement == 3: steer_input = -1  # Left
        if movement == 4: steer_input = 1   # Right

        self.player_y_vel += steer_input * self.STEER_FORCE
        self.player_y_vel = np.clip(self.player_y_vel, -self.MAX_Y_VEL, self.MAX_Y_VEL)
        self.player_y += self.player_y_vel
        self.player_y_vel *= self.Y_DRAG

        if space_held:
            self.player_speed += self.ACCELERATION
            # Add acceleration particles
            if self.np_random.random() < 0.7:
                self.particles.append(Particle(
                    pos=[self.SCREEN_WIDTH/2 - 10, self.player_y],
                    vel=[-self.player_speed / 2, self.np_random.uniform(-1, 1)],
                    color=self.COLOR_TRAIL,
                    radius=self.np_random.uniform(2, 5),
                    lifespan=10
                ))
        if shift_held:
            self.player_speed -= self.BRAKE_FORCE
        
        self.player_speed *= self.DRAG
        self.player_speed = np.clip(self.player_speed, self.MIN_SPEED, self.MAX_SPEED)
        
        self.track_progress += self.player_speed
        self.time_left -= 1.0 / self.FPS
        self.steps += 1

        # --- Update particles ---
        self.particles = [p for p in self.particles if p.update()]

        # --- Calculate reward and check termination ---
        reward = 0.1  # Survival reward
        
        # Near-miss penalty
        wall_top, wall_bottom = self._get_walls_at(self.track_progress)
        dist_to_walls = min(self.player_y - wall_top, wall_bottom - self.player_y)
        if 0 < dist_to_walls < self.NEAR_MISS_THRESHOLD:
            reward = -5.0

        terminated = False
        step_event_reward = 0

        # 1. Wall Collision
        if not (wall_top < self.player_y < wall_bottom):
            self.game_over = True
            terminated = True
            step_event_reward = -100
            # Add collision sparks
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 10)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append(Particle(
                    pos=[self.SCREEN_WIDTH/2, self.player_y], vel=vel,
                    color=self.COLOR_SPARKS, radius=self.np_random.uniform(1, 3), lifespan=15
                ))

        # 2. Time Out
        elif self.time_left <= 0:
            self.game_over = True
            terminated = True
            step_event_reward = -100

        # 3. Track Completion
        elif self.track_progress >= self.TRACK_LENGTH:
            step_event_reward = 100
            self.score += 100
            self.current_track += 1
            if self.current_track > self.TOTAL_TRACKS:
                self.game_over = True
                self.victory = True
                terminated = True
                step_event_reward += 300 # Final victory bonus
                self.score += 300
            else:
                self._setup_track() # Go to next track

        reward += step_event_reward
        self.score += reward # Update score with survival and penalties

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_walls_at(self, progress):
        """Get the y-coordinates of the top and bottom walls at a given progress."""
        if progress < 0 or progress >= self.TRACK_LENGTH:
            # Provide safe walls outside the track to prevent index errors
            center_y = self.SCREEN_HEIGHT / 2
            return center_y - self.TRACK_WIDTH / 2, center_y + self.TRACK_WIDTH / 2
            
        segment_len = self.TRACK_LENGTH / self.TRACK_POINTS
        idx = int(progress / segment_len)
        interp = (progress % segment_len) / segment_len
        
        p1 = self.track_centerline[idx]
        p2 = self.track_centerline[idx + 1]
        
        center_y = p1[1] + (p2[1] - p1[1]) * interp
        return center_y - self.TRACK_WIDTH / 2, center_y + self.TRACK_WIDTH / 2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_track()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self):
        # Smooth camera follow
        self.camera_x += (self.track_progress - self.camera_x) * 0.1

        points_top, points_bottom = [], []
        
        # Determine which track segments are visible
        segment_len = self.TRACK_LENGTH / self.TRACK_POINTS
        start_idx = max(0, int((self.camera_x - self.SCREEN_WIDTH) / segment_len))
        end_idx = min(self.TRACK_POINTS, int((self.camera_x + self.SCREEN_WIDTH) / segment_len) + 2)

        for i in range(start_idx, end_idx):
            world_x, center_y = self.track_centerline[i]
            screen_x = int(world_x - self.camera_x + self.SCREEN_WIDTH / 2)
            
            points_top.append((screen_x, int(center_y - self.TRACK_WIDTH / 2)))
            points_bottom.append((screen_x, int(center_y + self.TRACK_WIDTH / 2)))

        if len(points_top) > 1:
            # Draw glow effect first
            pygame.draw.aalines(self.screen, self.COLOR_TRACK_GLOW, False, points_top, 5)
            pygame.draw.aalines(self.screen, self.COLOR_TRACK_GLOW, False, points_bottom, 5)
            # Draw main track lines
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_top)
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_bottom)

        # Render finish line
        finish_x = self.TRACK_LENGTH - self.camera_x + self.SCREEN_WIDTH / 2
        if self.SCREEN_WIDTH > finish_x > 0:
            wall_top, wall_bottom = self._get_walls_at(self.TRACK_LENGTH - 1)
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, wall_top), (finish_x, wall_bottom), 5)

    def _render_player(self):
        x, y = self.SCREEN_WIDTH / 2, self.player_y
        size = 12
        points = [
            (x + size, y),
            (x - size / 2, y - size / 1.5),
            (x - size / 2, y + size / 1.5),
        ]
        
        # Draw glow
        glow_points = [
            (x + size*1.5, y),
            (x - size, y - size),
            (x - size, y + size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

        # Draw main triangle
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self.camera_x - self.SCREEN_WIDTH / 2)

    def _render_ui(self):
        # Time and Track
        time_text = self.font_small.render(f"TIME: {max(0, self.time_left):.1f}", True, self.COLOR_UI)
        track_text = self.font_small.render(f"TRACK: {self.current_track}/{self.TOTAL_TRACKS}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(track_text, (self.SCREEN_WIDTH - track_text.get_width() - 10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, self.SCREEN_HEIGHT - 30))

        # Speed Bar
        speed_ratio = (self.player_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        speed_ratio = np.clip(speed_ratio, 0, 1)
        bar_color = [int(c1 + (c2 - c1) * speed_ratio) for c1, c2 in zip(self.COLOR_SPEED_LOW, self.COLOR_SPEED_HIGH)]
        bar_width = self.SCREEN_WIDTH / 2
        bar_height = 10
        current_width = bar_width * speed_ratio
        pygame.draw.rect(self.screen, (50, 50, 50), (self.SCREEN_WIDTH/4, self.SCREEN_HEIGHT - 50, bar_width, bar_height))
        if current_width > 0:
            pygame.draw.rect(self.screen, bar_color, (self.SCREEN_WIDTH/4, self.SCREEN_HEIGHT - 50, current_width, bar_height))

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            msg_render = self.font_large.render(message, True, self.COLOR_UI)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH/2 - msg_render.get_width()/2, self.SCREEN_HEIGHT/2 - msg_render.get_height()/2 - 20))
            
            final_score_render = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI)
            self.screen.blit(final_score_render, (self.SCREEN_WIDTH/2 - final_score_render.get_width()/2, self.SCREEN_HEIGHT/2 + 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_track": self.current_track,
            "time_left": self.time_left,
            "player_speed": self.player_speed,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, pos, vel, color, radius, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.radius = radius
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface, camera_x_offset):
        alpha = int(255 * (self.lifespan / self.initial_lifespan))
        if alpha <= 0: return

        # Particles are in world space relative to the player's screen position
        # We don't need to adjust for camera, they are visual effects.
        pos_on_screen = (int(self.pos[0]), int(self.pos[1]))
        
        # Use a temporary surface for alpha blending
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.color + (alpha,), (self.radius, self.radius), self.radius)
        surface.blit(temp_surf, (pos_on_screen[0] - self.radius, pos_on_screen[1] - self.radius))

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a window to display the game
    pygame.display.set_caption("Line Racer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    pygame.quit()