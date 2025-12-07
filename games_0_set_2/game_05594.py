import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to move vertically. The car moves forward automatically. "
        "Hold Shift while turning (no-op action) to drift and earn bonus points."
    )

    game_description = (
        "A fast-paced, retro-style arcade racer. Steer your car on a procedural track, "
        "dodge obstacles, and drift through corners to maximize your score. Complete 3 laps to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Headless Pygame Setup ---
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 4500 # Increased to allow for 3 laps at base speed

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TRACK = (40, 50, 80)
        self.COLOR_TRACK_BORDER = (100, 110, 140)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 80, 80, 50)
        self.COLOR_OBSTACLE = (255, 255, 100)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_CHECKPOINT = (100, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Player
        self.PLAYER_X_POS = self.W // 4
        self.PLAYER_V_ACCEL = 1.5
        self.PLAYER_V_FRICTION = 0.85
        self.PLAYER_MAX_V = 8

        # World
        self.FORWARD_SPEED = 6
        self.LAP_LENGTH = 6000 # pixels
        self.TRACK_WIDTH = 150
        self.TRACK_ROUGHNESS = 0.4
        self.TRACK_MIN_Y = 100
        self.TRACK_MAX_Y = self.H - 100
        self.TRACK_SEGMENT_LENGTH = 50

        # Obstacles
        self.OBSTACLE_SPAWN_INTERVAL = 60 # steps
        self.OBSTACLE_BASE_SPEED = 1.0
        self.OBSTACLE_MAX_SPEED = 3.0
        self.OBSTACLE_SPEED_INCREASE_INTERVAL = 500
        self.OBSTACLE_SPEED_INCREASE_AMOUNT = 0.05
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Surfaces
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Internal State - initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_pos = None
        self.player_vel = None
        self.is_drifting = False
        self.drift_duration = 0
        
        self.world_x_offset = 0
        self.laps_completed = 0
        self.lap_time_steps = 0
        
        self.track_points = None
        self.obstacles = None
        self.particles = None
        self.speed_lines = None
        
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self.next_obstacle_spawn = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.H / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_drifting = False
        self.drift_duration = 0

        self.world_x_offset = 0
        self.laps_completed = 0
        self.lap_time_steps = 0

        self.obstacles = deque()
        self.particles = deque()
        self.speed_lines = deque()
        
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self.next_obstacle_spawn = 0

        self._generate_initial_track()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.lap_time_steps += 1
        reward = 0.1  # Survival reward

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Vertical movement
        if movement == 1:  # Up
            self.player_vel.y -= self.PLAYER_V_ACCEL
        elif movement == 2:  # Down
            self.player_vel.y += self.PLAYER_V_ACCEL
        
        # Drifting logic
        was_drifting = self.is_drifting
        self.is_drifting = shift_held

        if self.is_drifting:
            self.drift_duration += 1
            reward += 0.05 # Small reward for maintaining drift
            if self.drift_duration % 3 == 0:
                # sound: drift_screech.wav
                self._create_particles(self.player_pos, 1, (2, 4), (-2, 2), 15)
        elif was_drifting and not self.is_drifting:
            # Reward for completing a drift
            drift_bonus = 1 + int(self.drift_duration / self.FPS) # +1 per second of drift
            reward += drift_bonus
            self.score += drift_bonus
            self.drift_duration = 0

        # --- Physics & World Update ---
        self.player_vel.y *= self.PLAYER_V_FRICTION
        self.player_vel.y = np.clip(self.player_vel.y, -self.PLAYER_MAX_V, self.PLAYER_MAX_V)
        self.player_pos.y += self.player_vel.y
        
        self.world_x_offset += self.FORWARD_SPEED

        # --- Track & Player Clamping ---
        track_y_at_player, track_half_width = self._get_track_state_at(self.player_pos.x)
        track_top = track_y_at_player - track_half_width
        track_bottom = track_y_at_player + track_half_width
        
        self.player_pos.y = np.clip(self.player_pos.y, track_top, track_bottom)

        # --- Dynamic Elements Update ---
        self._update_track()
        self._update_obstacles()
        self._update_particles()
        self._update_speed_lines()
        self._spawn_obstacles()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_current_speed = min(self.OBSTACLE_MAX_SPEED, self.obstacle_current_speed + self.OBSTACLE_SPEED_INCREASE_AMOUNT)

        # --- Collision & Termination Check ---
        terminated = False
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 5, 20, 10)
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                # sound: explosion.wav
                self.game_over = True
                reward = -5
                self.score -= 5
                break
        
        if not self.game_over:
            # Lap completion
            if self.world_x_offset >= (self.laps_completed + 1) * self.LAP_LENGTH:
                # sound: lap_complete.wav
                self.laps_completed += 1
                self.lap_time_steps = 0
                reward += 50
                self.score += 50
                if self.laps_completed >= 3:
                    # sound: victory.wav
                    self.game_over = True
                    reward += 100
                    self.score += 100
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            truncated = False

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Generation & Update Helpers ---

    def _generate_initial_track(self):
        self.track_points = deque()
        y = self.H / 2
        for x in range(-self.TRACK_SEGMENT_LENGTH * 2, self.W + self.TRACK_SEGMENT_LENGTH * 2, self.TRACK_SEGMENT_LENGTH):
            y += self.np_random.uniform(-self.TRACK_SEGMENT_LENGTH * self.TRACK_ROUGHNESS, self.TRACK_SEGMENT_LENGTH * self.TRACK_ROUGHNESS)
            y = np.clip(y, self.TRACK_MIN_Y, self.TRACK_MAX_Y)
            self.track_points.append(pygame.Vector2(x, y))

    def _update_track(self):
        # Remove points that are off-screen to the left
        while self.track_points and self.track_points[0].x - self.world_x_offset < -self.TRACK_SEGMENT_LENGTH * 2:
            self.track_points.popleft()
        
        # Add new points to the right
        while self.track_points[-1].x - self.world_x_offset < self.W + self.TRACK_SEGMENT_LENGTH * 2:
            last_point = self.track_points[-1]
            new_x = last_point.x + self.TRACK_SEGMENT_LENGTH
            new_y = last_point.y + self.np_random.uniform(-self.TRACK_SEGMENT_LENGTH * self.TRACK_ROUGHNESS, self.TRACK_SEGMENT_LENGTH * self.TRACK_ROUGHNESS)
            new_y = np.clip(new_y, self.TRACK_MIN_Y, self.TRACK_MAX_Y)
            self.track_points.append(pygame.Vector2(new_x, new_y))

    def _get_track_state_at(self, world_x_coord):
        # Find the two track points surrounding the x coordinate
        p1, p2 = None, None
        for i in range(len(self.track_points) - 1):
            if self.track_points[i].x <= world_x_coord + self.world_x_offset < self.track_points[i+1].x:
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                break
        
        if p1 is None or p2 is None: # Fallback
            return self.H / 2, self.TRACK_WIDTH / 2

        # Interpolate between the two points
        interp_factor = (world_x_coord + self.world_x_offset - p1.x) / (p2.x - p1.x)
        center_y = p1.y + (p2.y - p1.y) * interp_factor
        
        return center_y, self.TRACK_WIDTH / 2

    def _spawn_obstacles(self):
        if self.steps > self.next_obstacle_spawn:
            track_y, track_half_width = self._get_track_state_at(self.W)
            
            obs_y = self.np_random.uniform(track_y - track_half_width + 10, track_y + track_half_width - 10)
            obs_size = self.np_random.integers(15, 25)
            
            obstacle = {
                'rect': pygame.Rect(self.W, obs_y - obs_size/2, obs_size, obs_size),
                'vx': -self.obstacle_current_speed * self.np_random.uniform(0.8, 1.2)
            }
            self.obstacles.append(obstacle)
            self.next_obstacle_spawn = self.steps + self.np_random.integers(int(self.OBSTACLE_SPAWN_INTERVAL * 0.8), int(self.OBSTACLE_SPAWN_INTERVAL * 1.2))

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['rect'].x += obs['vx']
        self.obstacles = deque(o for o in self.obstacles if o['rect'].right > 0)

    def _create_particles(self, pos, count, speed_range, angle_range, life):
        for _ in range(count):
            angle = self.np_random.uniform(angle_range[0], angle_range[1])
            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            vel = pygame.Vector2(speed, 0).rotate(math.degrees(angle))
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = deque(p for p in self.particles if p['life'] > 0)

    def _update_speed_lines(self):
        if self.np_random.random() < 0.5:
            y = self.np_random.integers(0, self.H)
            self.speed_lines.append({'y': y, 'life': 10, 'max_life': 10, 'len': self.np_random.integers(20, 50)})
        
        for line in self.speed_lines:
            line['life'] -= 1
        self.speed_lines = deque(l for l in self.speed_lines if l['life'] > 0)

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_track()
        self._render_checkpoints()
        self._render_obstacles()
        self._render_particles()
        self._render_speed_lines()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self):
        points_top = []
        points_bottom = []
        for p in self.track_points:
            screen_x = p.x - self.world_x_offset
            points_top.append((screen_x, p.y - self.TRACK_WIDTH / 2))
            points_bottom.append((screen_x, p.y + self.TRACK_WIDTH / 2))
        
        if len(points_top) > 1:
            track_poly_points = points_top + points_bottom[::-1]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x,y in track_poly_points], self.COLOR_TRACK)
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x,y in track_poly_points], self.COLOR_TRACK_BORDER)

    def _render_checkpoints(self):
        for i in range(self.laps_completed, 4):
            checkpoint_x = i * self.LAP_LENGTH - self.world_x_offset
            if 0 < checkpoint_x < self.W:
                track_y, track_half_width = self._get_track_state_at(checkpoint_x - self.PLAYER_X_POS + self.player_pos.x)
                pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (int(checkpoint_x), int(track_y - track_half_width)), (int(checkpoint_x), int(track_y + track_half_width)), 5)

    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_PARTICLE, alpha), (size, size), size)
                self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_speed_lines(self):
        for line in self.speed_lines:
            alpha = int(100 * (line['life'] / line['max_life']))
            color = (*self.COLOR_PARTICLE, alpha)
            start_x = self.W
            end_x = self.W - line['len']
            pygame.draw.line(self.screen, color, (start_x, line['y']), (end_x, line['y']), 1)

    def _render_player(self):
        # Car body
        car_rect = pygame.Rect(0, 0, 24, 12)
        car_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Drift glow
        if self.is_drifting:
            glow_radius = 15 + (self.drift_duration % 10) * 0.3
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (car_rect.centerx - glow_radius, car_rect.centery - glow_radius))
            
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, car_rect, border_radius=3)
        # Windshield
        pygame.draw.rect(self.screen, self.COLOR_BG, (car_rect.x + 12, car_rect.y + 2, 8, 8), border_radius=2)


    def _render_ui(self):
        # Lap Counter
        lap_text = f"LAP: {min(3, self.laps_completed + 1)}/3"
        self._draw_text(lap_text, (10, 10))

        # Lap Time
        lap_time_sec = self.lap_time_steps / self.FPS
        time_text = f"TIME: {lap_time_sec:.2f}"
        self._draw_text(time_text, (self.W - 150, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.W / 2, 10), center=True)
        
        # Drift Multiplier
        if self.is_drifting and self.drift_duration > self.FPS / 2:
            drift_mult = 1 + int(self.drift_duration / self.FPS)
            drift_text = f"DRIFT x{drift_mult}"
            self._draw_text(drift_text, (self.W / 2, self.H - 40), center=True, font=self.font_big)

        if self.game_over:
            if self.laps_completed >= 3:
                end_text = "FINISH!"
            else:
                end_text = "CRASHED!"
            self._draw_text(end_text, (self.W/2, self.H/2 - 30), center=True, font=self.font_big)


    def _draw_text(self, text, pos, color=None, font=None, center=False):
        if color is None: color = self.COLOR_TEXT
        if font is None: font = self.font_ui
        
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        # Add a semi-transparent background for readability
        bg_rect = text_rect.inflate(10, 6)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(bg_surf, bg_rect)

        self.screen.blit(text_surface, text_rect)

    # --- Gymnasium Boilerplate ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps_completed,
            "lap_time": self.lap_time_steps / self.FPS if self.FPS > 0 else 0,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display, comment out the next line
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- Manual Play ---
    # This part requires a window. Comment out if running headlessly.
    render_human = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    if render_human:
        # Re-initialize pygame with video
        pygame.quit()
        pygame.init()
        display_screen = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption("Arcade Racer")
    
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Action mapping for human players
        action = [0, 0, 0] # Default no-op
        if render_human:
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
        else: # Sample actions if running headlessly
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode End. Final Score: {info['score']:.2f}, Laps: {info['laps']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0


        if render_human:
            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)

    env.close()