import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Hold Shift to drift and press Space for a speed boost."
    )

    game_description = (
        "A fast-paced, side-scrolling retro-futuristic racer. Navigate a procedurally generated neon track, manage your boosts, and set the best time over 3 laps."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TRACK_LENGTH = 15000  # Pixels per lap
        self.NUM_LAPS = 3
        self.LAP_TIME_LIMIT = 60 # seconds

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors (Neon retro)
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_TRACK = (0, 255, 128)
        self.COLOR_PLAYER = (255, 20, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 120)
        self.COLOR_BOOST = (0, 200, 255)
        self.COLOR_DRIFT = (220, 220, 220)
        self.COLOR_FINISH_LINE_1 = (255, 255, 0)
        self.COLOR_FINISH_LINE_2 = (200, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.camera_x = None
        self.track = None
        self.track_gen_pos = None
        self.particles = None
        
        self.steps = None
        self.score = None
        self.lap = None
        self.boosts = None
        self.lap_start_step = None
        self.total_steps = None
        
        self.is_boosting = None
        self.boost_timer = None
        self.is_drifting = None
        
        self.game_over = None
        self.game_over_message = None

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([100.0, self.HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = 0.0

        # World state
        self.camera_x = 0.0
        self.track = deque()
        self.track_gen_pos = 0.0
        self._generate_initial_track()
        self.particles = []

        # Game state
        self.steps = 0
        self.score = 0
        self.lap = 1
        self.boosts = 3
        self.lap_start_step = 0
        self.total_steps = 0
        
        self.is_boosting = False
        self.boost_timer = 0
        self.is_drifting = False
        
        self.game_over = False
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over
        truncated = False

        if not terminated:
            self.steps += 1
            self.total_steps += 1
            
            reward += self._update_game_state(action)
            
            self._update_world()
            
            collision_penalty, did_crash = self._handle_collisions()
            if did_crash:
                reward += collision_penalty
                self.game_over = True
                self.game_over_message = "CRASHED!"
                terminated = True
            
            if not terminated:
                lap_reward, lap_completed = self._check_lap_completion()
                if lap_completed:
                    reward += lap_reward
                    if self.lap > self.NUM_LAPS:
                        finish_reward = 50 * max(0, (self.NUM_LAPS * self.LAP_TIME_LIMIT - self.total_steps / self.FPS) / (self.NUM_LAPS * self.LAP_TIME_LIMIT))
                        reward += finish_reward
                        self.game_over = True
                        self.game_over_message = "RACE COMPLETE!"
                        terminated = True

            if not terminated and (self.total_steps - self.lap_start_step) / self.FPS > self.LAP_TIME_LIMIT:
                reward -= 10
                self.game_over = True
                self.game_over_message = "TIME'S UP!"
                terminated = True
                
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_game_state(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Physics Constants ---
        ACCEL = 0.15
        BRAKE = 0.2
        STEER_FORCE = 0.5
        BOOST_FORCE = 1.5
        BOOST_DURATION = 15 # steps
        X_FRICTION = 0.98
        Y_FRICTION = 0.90
        MIN_SPEED = 2.0
        MAX_SPEED = 10.0
        
        # --- Drifting Physics ---
        self.is_drifting = shift_held and abs(self.player_vel[1]) > 0.5
        drift_steer_mod = 2.0 if self.is_drifting else 1.0
        drift_y_friction = 0.98 if self.is_drifting else Y_FRICTION

        # --- Player Input ---
        # Acceleration / Braking
        if movement == 1: # Up
            self.player_vel[0] += ACCEL
        elif movement == 2: # Down
            self.player_vel[0] -= BRAKE
        
        # Steering
        if movement == 3: # Left
            self.player_vel[1] -= STEER_FORCE * drift_steer_mod
        if movement == 4: # Right
            self.player_vel[1] += STEER_FORCE * drift_steer_mod

        # Boost
        if space_held and self.boosts > 0 and not self.is_boosting:
            self.is_boosting = True
            self.boost_timer = BOOST_DURATION
            self.boosts -= 1
            
        if self.is_boosting:
            self.player_vel[0] += BOOST_FORCE
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.is_boosting = False

        # --- Physics Update ---
        # Apply friction and clamp speed
        self.player_vel[0] *= X_FRICTION
        self.player_vel[1] *= drift_y_friction
        self.player_vel[0] = max(MIN_SPEED, self.player_vel[0])
        self.player_vel[0] = min(MAX_SPEED, self.player_vel[0])
        
        # Update position
        self.player_pos += self.player_vel
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # Update camera
        self.camera_x += self.player_vel[0]

        # Update angle for visual flair
        self.player_angle = self.player_vel[1] * 2.0

        # --- Particle Effects ---
        if self.is_boosting:
            for _ in range(3):
                p_vel = np.array([-self.player_vel[0] * 1.5, random.uniform(-1, 1)])
                self._create_particle(self.player_pos, p_vel, self.COLOR_BOOST, 20, 4)
        if self.is_drifting:
            p_vel = np.array([-self.player_vel[0] * 0.5, random.uniform(-2, 2)])
            self._create_particle(self.player_pos, p_vel, self.COLOR_DRIFT, 15, 3)

        # Continuous reward for forward progress
        return self.player_vel[0] * 0.01

    def _update_world(self):
        # Generate new track segments
        while self.track_gen_pos < self.camera_x + self.WIDTH + 100:
            self._generate_track_segment()

        # Prune old track segments
        while self.track and self.track[0][0] < self.camera_x - 50:
            self.track.popleft()
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _handle_collisions(self):
        # Find current track boundaries at player's x position
        try:
            player_world_x = self.camera_x + self.player_pos[0]
            
            # Find the track segment the player is on
            p1 = None
            for seg in self.track:
                if seg[0] >= player_world_x:
                    p1 = seg
                    break
            if p1 is None:
                return 0, True # Off the end of the track

            p1_index = self.track.index(p1)
            if p1_index == 0:
                 return 0, True # Off the start of the track

            p0 = self.track[p1_index - 1]

        except (ValueError, IndexError):
            return 0, True # Off the generated track, crash

        # Linear interpolation
        if (p1[0] - p0[0]) == 0: # Avoid division by zero
             return 0, True

        t = (player_world_x - p0[0]) / (p1[0] - p0[0])
        current_top = p0[1] + t * (p1[1] - p0[1])
        current_bottom = p0[2] + t * (p1[2] - p0[2])

        player_half_height = 10 # approximate
        if not (current_top + player_half_height < self.player_pos[1] < current_bottom - player_half_height):
            self.player_vel *= 0.1 # Kill velocity
            return -20, True # Terminal crash
        
        return 0, False

    def _check_lap_completion(self):
        if self.camera_x >= self.lap * self.TRACK_LENGTH:
            self.lap += 1
            self.lap_start_step = self.total_steps
            # Give a boost back for completing a lap
            self.boosts = min(3, self.boosts + 1)
            return 10, True
        return 0, False

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
            "lap": self.lap,
            "total_time": self.total_steps / self.FPS,
            "boosts": self.boosts,
        }

    def _render_game(self):
        # Draw parallax stars
        for i in range(50):
            seed = i * 12345
            x = (self.WIDTH * (hash(seed) % 1000) / 1000)
            y = (self.HEIGHT * (hash(seed*2) % 1000) / 1000)
            # Slower layer
            px1 = int((x - self.camera_x * 0.1) % self.WIDTH)
            pygame.gfxdraw.pixel(self.screen, px1, int(y), (100, 100, 120))
            # Faster layer
            px2 = int((x - self.camera_x * 0.3) % self.WIDTH)
            pygame.gfxdraw.pixel(self.screen, px2, int(y), (180, 180, 200))

        # Draw track
        if len(self.track) > 1:
            points_top = []
            points_bottom = []
            for p in self.track:
                draw_x = int(p[0] - self.camera_x)
                if -10 < draw_x < self.WIDTH + 10:
                    points_top.append((draw_x, p[1]))
                    points_bottom.append((draw_x, p[2]))
            
            if len(points_top) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_top)
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_bottom)


        # Draw finish lines
        for l in range(self.NUM_LAPS + 2):
            finish_x = l * self.TRACK_LENGTH - self.camera_x
            if -50 < finish_x < self.WIDTH + 50:
                self._draw_finish_line(finish_x)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            
            # Manual alpha blending to create a 3-component RGB color.
            # This avoids a TypeError with gfxdraw.filled_circle on some systems
            # when using a 4-component RGBA color.
            alpha_ratio = max(0.0, min(1.0, p['life'] / p['max_life']))
            fg_color = p['color']
            bg_color = self.COLOR_BG
            
            r = int(fg_color[0] * alpha_ratio + bg_color[0] * (1 - alpha_ratio))
            g = int(fg_color[1] * alpha_ratio + bg_color[1] * (1 - alpha_ratio))
            b = int(fg_color[2] * alpha_ratio + bg_color[2] * (1 - alpha_ratio))
            
            blended_color = (r, g, b)

            if p['size'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), blended_color)

        # Draw player
        self._draw_player()

    def _draw_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        angle_rad = math.radians(self.player_angle)
        
        # Kart body points
        points = [
            (-12, -6), (12, 0), (-12, 6), (-8, 0)
        ]
        
        # Rotate and translate points
        rotated_points = []
        for p_x, p_y in points:
            new_x = p_x * math.cos(angle_rad) - p_y * math.sin(angle_rad)
            new_y = p_x * math.sin(angle_rad) + p_y * math.cos(angle_rad)
            rotated_points.append((x + new_x, y + new_y))

        # Draw glow
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, (*self.COLOR_PLAYER_GLOW, 50))
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, (*self.COLOR_PLAYER_GLOW, 80))
        
        # Draw main body
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Lap Time
        lap_time = (self.total_steps - self.lap_start_step) / self.FPS
        time_text = f"TIME: {lap_time:.2f}s"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Boosts
        boost_text = f"BOOSTS: {self.boosts}"
        boost_surf = self.font_small.render(boost_text, True, self.COLOR_TEXT)
        self.screen.blit(boost_surf, (self.WIDTH - boost_surf.get_width() - 10, 10))
        
        # Current Lap
        lap_text = f"LAP {min(self.lap, self.NUM_LAPS)} / {self.NUM_LAPS}"
        lap_surf = self.font_large.render(lap_text, True, self.COLOR_TEXT)
        lap_rect = lap_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(lap_surf, lap_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _generate_initial_track(self):
        y_center = self.HEIGHT / 2.0
        track_width = 150.0
        for i in range(int(self.WIDTH / 20) + 10):
            self.track.append((i * 20.0, y_center - track_width / 2, y_center + track_width / 2))
        self.track_gen_pos = self.track[-1][0]

    def _generate_track_segment(self):
        # Difficulty scaling: curvature increases with lap
        max_turn_angle = 10 + 5 * (self.lap - 1)
        
        last_x, last_top, last_bot = self.track[-1]
        last_center = (last_top + last_bot) / 2.0
        
        new_x = last_x + 20
        
        # Random walk for the center line
        change = random.uniform(-max_turn_angle, max_turn_angle)
        new_center = last_center + change
        
        # Keep track within reasonable bounds
        track_width = last_bot - last_top
        new_center = np.clip(new_center, track_width, self.HEIGHT - track_width)
        
        new_top = new_center - track_width / 2
        new_bot = new_center + track_width / 2
        
        self.track.append((new_x, new_top, new_bot))
        self.track_gen_pos = new_x

    def _create_particle(self, pos, vel, color, life, size):
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'color': color,
            'life': life + random.uniform(-life*0.2, life*0.2),
            'max_life': life,
            'size': size
        })

    def _draw_finish_line(self, x_pos):
        if not self.track:
            return
        try:
            p1 = None
            for seg in self.track:
                if seg[0] >= x_pos:
                    p1 = seg
                    break
            if p1 is None: return

            p1_index = self.track.index(p1)
            if p1_index == 0: return
            
            p0 = self.track[p1_index - 1]
            if (p1[0] - p0[0]) == 0: return

            t = (x_pos - p0[0]) / (p1[0] - p0[0])
            y_top = p0[1] + t * (p1[1] - p0[1])
            y_bot = p0[2] + t * (p1[2] - p0[2])
        except (ValueError, IndexError, ZeroDivisionError):
            return

        check_height = 10
        for i, y in enumerate(np.arange(y_top, y_bot, check_height)):
            color = self.COLOR_FINISH_LINE_1 if i % 2 == 0 else self.COLOR_FINISH_LINE_2
            pygame.draw.rect(self.screen, color, (int(x_pos), int(y), 10, check_height))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    
    # To map keyboard inputs to actions for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Re-initialize pygame with a display for human play
    pygame.display.init()
    pygame.display.set_caption("Retro Racer")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key in map if multiple are pressed
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()


    print(f"Game Over! Final Info: {info}")
    env.close()