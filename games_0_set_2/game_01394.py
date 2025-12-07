
# Generated: 2025-08-27T16:59:43.547545
# Source Brief: brief_01394.md
# Brief Index: 1394

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon (use boost)."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use them to reach the finish line against the clock."
    )

    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 20
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_TRACK = (100, 100, 120)
    COLOR_TRACK_BORDER = (200, 200, 220)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_OBSTACLE_GLOW = (100, 200, 255)
    COLOR_BOOST = (50, 255, 150)
    COLOR_BOOST_GLOW = (150, 255, 200)
    COLOR_FINISH_DARK = (40, 40, 50)
    COLOR_FINISH_LIGHT = (220, 220, 230)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)

    # Physics
    ACCELERATION = 0.15
    BRAKING = 0.3
    MAX_SPEED = 8.0
    TURN_SPEED = 0.08
    FRICTION = 0.97  # Multiplier
    DRIFT_FRICTION = 0.99
    DRIFT_TURN_MOD = 1.5
    BOOST_POWER = 15.0

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 30)
        
        self.render_mode = render_mode
        self.game_over = False
        
        # Initialize state variables in reset
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.boost_charges = 0
        self.obstacles = []
        self.boost_pads = []
        self.particles = []
        self.track_path = []
        self.total_track_length = 0
        self.last_dist_to_finish = 0
        self.rng = None

        self.validate_implementation()

    def _generate_track(self):
        self.track_path = [
            pygame.Vector2(200, 800),
            pygame.Vector2(800, 800),
            pygame.Vector2(1200, 400),
            pygame.Vector2(1100, -200),
            pygame.Vector2(600, -600),
            pygame.Vector2(-200, -600),
            pygame.Vector2(-800, -200),
            pygame.Vector2(-800, 400),
            pygame.Vector2(-200, 800),
            pygame.Vector2(200, 800) # Connect back for length calculation
        ]
        self.total_track_length = sum((self.track_path[i] - self.track_path[i+1]).length() for i in range(len(self.track_path)-1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self._generate_track()

        self.player_pos = self.track_path[0].copy() + pygame.Vector2(0, -50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -math.pi / 2  # Pointing "up" the screen initially
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.boost_charges = 1

        self.obstacles = self._create_obstacles()
        self.boost_pads = self._create_boost_pads()
        self.particles = []
        
        self.last_dist_to_finish = self._get_dist_to_finish()

        return self._get_observation(), self._get_info()

    def _create_obstacles(self):
        obstacles = []
        # Obstacle moving across the first straight
        obstacles.append({'pos': pygame.Vector2(400, 750), 'start': pygame.Vector2(400, 750), 'end': pygame.Vector2(600, 750), 'speed': 1.0, 'size': 15, 't': self.rng.random()})
        # Obstacle on the big curve
        obstacles.append({'pos': pygame.Vector2(1250, 100), 'start': pygame.Vector2(1250, 100), 'end': pygame.Vector2(1050, 100), 'speed': 1.5, 'size': 20, 't': self.rng.random()})
        # Obstacle on the top straight
        obstacles.append({'pos': pygame.Vector2(0, -650), 'start': pygame.Vector2(0, -650), 'end': pygame.Vector2(400, -650), 'speed': 2.0, 'size': 15, 't': self.rng.random()})
        # Obstacle on the final straight
        obstacles.append({'pos': pygame.Vector2(-500, 600), 'start': pygame.Vector2(-500, 600), 'end': pygame.Vector2(-500, 200), 'speed': 1.8, 'size': 18, 't': self.rng.random()})
        return obstacles

    def _create_boost_pads(self):
        return [
            {'pos': pygame.Vector2(1000, 600), 'size': 20, 'collected': False},
            {'pos': pygame.Vector2(800, -600), 'size': 20, 'collected': False},
            {'pos': pygame.Vector2(-800, 0), 'size': 20, 'collected': False},
        ]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # In auto_advance mode, ensure consistent timing
        self.clock.tick(self.FPS)
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        
        self._update_player(movement, space_held, shift_held)
        self._update_obstacles()
        self._update_particles()
        
        # --- Rewards and Termination ---
        reward = 0
        terminated = False

        # Progress reward
        current_dist_to_finish = self._get_dist_to_finish()
        progress = self.last_dist_to_finish - current_dist_to_finish
        reward += progress * 0.05  # Scaled reward for progress
        self.last_dist_to_finish = current_dist_to_finish
        
        # Collision checks
        for obstacle in self.obstacles:
            if self.player_pos.distance_to(obstacle['pos']) < 10 + obstacle['size']:
                # sfx: explosion
                reward = -100
                terminated = True
                self.game_over = True
                break
        
        for boost in self.boost_pads:
            if not boost['collected'] and self.player_pos.distance_to(boost['pos']) < 10 + boost['size']:
                # sfx: boost_pickup
                boost['collected'] = True
                self.boost_charges += 1
                reward += 5
        
        # Check finish line
        finish_line_start = self.track_path[0] + pygame.Vector2(0, 50)
        finish_line_end = self.track_path[0] + pygame.Vector2(0, -50)
        if self.player_pos.y < finish_line_start.y and self.player_pos.x > 0 and self.last_dist_to_finish < 200:
             if self._dist_to_line_segment(self.player_pos, finish_line_start, finish_line_end) < 50:
                # sfx: win_jingle
                reward = 100
                terminated = True
                self.game_over = True

        # Time limit
        if self.time_left <= 0:
            # sfx: lose_sound
            if not terminated: # Avoid double penalty
                reward = -100
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, space_held, shift_held):
        # --- Turning ---
        turn_mod = self.DRIFT_TURN_MOD if shift_held else 1.0
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED * turn_mod
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED * turn_mod

        # --- Acceleration & Braking ---
        if movement == 1:  # Up
            accel_vec = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.ACCELERATION
            self.player_vel += accel_vec
        if movement == 2:  # Down
            self.player_vel *= 0.9 # More effective braking than just friction
        
        # --- Boost ---
        if space_held and self.boost_charges > 0:
            # sfx: boost_activate
            self.boost_charges -= 1
            boost_vec = pygame.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.BOOST_POWER
            self.player_vel += boost_vec
            # Boost particles
            for _ in range(30):
                self._create_particle(self.player_pos, is_boost=True)

        # --- Physics ---
        friction = self.DRIFT_FRICTION if shift_held else self.FRICTION
        self.player_vel *= friction
        if self.player_vel.length() > self.MAX_SPEED:
            self.player_vel.scale_to_length(self.MAX_SPEED)
        
        self.player_pos += self.player_vel

        # --- Trail Particles ---
        is_drifting = shift_held and self.player_vel.length() > 2.0
        if self.player_vel.length() > 1.0:
            self._create_particle(self.player_pos, is_drifting=is_drifting)
            if is_drifting:
                self._create_particle(self.player_pos, is_drifting=is_drifting)


    def _update_obstacles(self):
        # Difficulty scaling
        speed_increase = (self.steps / 1000) * 0.05
        for obs in self.obstacles:
            obs['t'] = (obs['t'] + (obs['speed'] + speed_increase) * 0.01) % 2
            t_pingpong = 1 - abs(obs['t'] - 1)
            obs['pos'] = obs['start'].lerp(obs['end'], t_pingpong)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

    def _create_particle(self, pos, is_drifting=False, is_boost=False):
        angle = self.rng.uniform(-math.pi, math.pi)
        speed = self.rng.uniform(0.5, 2.0)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        life = 20 if is_drifting or is_boost else 15
        size = self.rng.uniform(3, 6)
        
        if is_boost:
            color = (self.rng.choice([255, 200, 150]), self.rng.choice([255, 100, 50]), 0)
            life = 40
            speed = self.rng.uniform(2.0, 5.0)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.rng.uniform(5, 10)
        elif is_drifting:
            color = (200, 200, 200)
        else:
            color = (150, 50, 50)
            
        self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'size': size, 'color': color})

    def _get_dist_to_finish(self):
        # Find the closest point on the track to the player
        min_dist = float('inf')
        closest_segment_index = 0
        
        for i in range(len(self.track_path) - 1):
            p1 = self.track_path[i]
            p2 = self.track_path[i+1]
            dist = self._dist_to_line_segment(self.player_pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_segment_index = i

        # Calculate distance from player's projection on the segment to the end of the segment
        p1 = self.track_path[closest_segment_index]
        p2 = self.track_path[closest_segment_index+1]
        
        v = p2 - p1
        w = self.player_pos - p1
        
        c1 = w.dot(v)
        if c1 <= 0: # Before start of segment
            dist_on_segment = (self.player_pos - p1).length()
        else:
            c2 = v.dot(v)
            if c2 <= c1: # After end of segment
                dist_on_segment = 0
            else:
                b = c1 / c2
                pb = p1 + b * v
                dist_on_segment = (p2 - pb).length()

        # Add lengths of all subsequent segments
        remaining_dist = dist_on_segment
        for i in range(closest_segment_index + 1, len(self.track_path) - 1):
            remaining_dist += (self.track_path[i] - self.track_path[i+1]).length()
            
        return remaining_dist

    def _dist_to_line_segment(self, p, v, w):
        l2 = (v - w).length_squared()
        if l2 == 0.0:
            return (p - v).length()
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)
        return (p - projection).length()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = self.player_pos.x - self.WIDTH / 2
        cam_y = self.player_pos.y - self.HEIGHT / 2

        # --- Render Track ---
        track_points_on_screen = [(p.x - cam_x, p.y - cam_y) for p in self.track_path]
        pygame.draw.aalines(self.screen, self.COLOR_TRACK_BORDER, False, track_points_on_screen, 50)
        pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, track_points_on_screen, 46)

        # --- Render Finish Line ---
        p_start = self.track_path[0]
        p_end = self.track_path[-1] # Same as start
        direction = (self.track_path[1] - p_start).normalize()
        perp_dir = pygame.Vector2(-direction.y, direction.x)
        
        for i in range(-2, 3):
            for j in range(-1, 1):
                color = self.COLOR_FINISH_LIGHT if (i + j) % 2 == 0 else self.COLOR_FINISH_DARK
                rect_center = p_start + perp_dir * i * 10 + direction * j * 10
                points = [
                    rect_center + perp_dir*5 + direction*5,
                    rect_center - perp_dir*5 + direction*5,
                    rect_center - perp_dir*5 - direction*5,
                    rect_center + perp_dir*5 - direction*5,
                ]
                screen_points = [(p.x - cam_x, p.y - cam_y) for p in points]
                pygame.gfxdraw.filled_polygon(self.screen, screen_points, color)
                pygame.gfxdraw.aapolygon(self.screen, screen_points, color)

        # --- Render Particles ---
        for p in self.particles:
            if p['size'] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x - cam_x), int(p['pos'].y - cam_y), int(p['size']), p['color'])

        # --- Render Boost Pads ---
        for boost in self.boost_pads:
            if not boost['collected']:
                bx, by = int(boost['pos'].x - cam_x), int(boost['pos'].y - cam_y)
                pulse = (math.sin(self.steps * 0.1) + 1) / 2
                glow_size = int(boost['size'] + pulse * 5)
                
                # Use alpha blending for glow
                glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_BOOST_GLOW, 50))
                pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, int(glow_size * 0.7), (*self.COLOR_BOOST_GLOW, 100))
                self.screen.blit(glow_surf, (bx - glow_size, by - glow_size))

                pygame.gfxdraw.filled_circle(self.screen, bx, by, boost['size'], self.COLOR_BOOST)
                pygame.gfxdraw.aacircle(self.screen, bx, by, boost['size'], self.COLOR_BOOST_GLOW)


        # --- Render Obstacles ---
        for obs in self.obstacles:
            ox, oy = int(obs['pos'].x - cam_x), int(obs['pos'].y - cam_y)
            glow_size = int(obs['size'] * 1.5)
            
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_OBSTACLE_GLOW, 50))
            self.screen.blit(glow_surf, (ox - glow_size, oy - glow_size))
            
            pygame.gfxdraw.filled_circle(self.screen, ox, oy, obs['size'], self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, ox, oy, obs['size'], self.COLOR_OBSTACLE_GLOW)

        # --- Render Player ---
        px, py = self.WIDTH / 2, self.HEIGHT / 2 # Player is always centered
        
        # Glow
        glow_size = 25
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, 80))
        self.screen.blit(glow_surf, (px - glow_size, py - glow_size))

        # Car body (triangle)
        p1 = pygame.Vector2(15, 0).rotate_rad(self.player_angle) + pygame.Vector2(px, py)
        p2 = pygame.Vector2(-7, 7).rotate_rad(self.player_angle) + pygame.Vector2(px, py)
        p3 = pygame.Vector2(-7, -7).rotate_rad(self.player_angle) + pygame.Vector2(px, py)
        
        player_points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Timer
        time_sec = self.time_left / self.FPS
        timer_text = f"{max(0, time_sec):.1f}"
        
        timer_color = self.COLOR_TEXT
        if time_sec < 10: timer_color = self.COLOR_TIMER_WARN
        if time_sec < 5: timer_color = self.COLOR_TIMER_CRIT
            
        text_surface = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(text_surface, (self.WIDTH // 2 - text_surface.get_width() // 2, 10))
        
        # Boosts
        boost_text = f"BOOSTS: {self.boost_charges}"
        boost_surface = self.font_small.render(boost_text, True, self.COLOR_BOOST if self.boost_charges > 0 else self.COLOR_TEXT)
        self.screen.blit(boost_surface, (self.WIDTH - boost_surface.get_width() - 15, 15))

        # Score (for debugging RL)
        # score_text = f"Score: {self.score:.1f}"
        # score_surface = self.font_small.render(score_text, True, self.COLOR_TEXT)
        # self.screen.blit(score_surface, (15, 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "boost_charges": self.boost_charges,
            "distance_to_finish": self.last_dist_to_finish,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset first to initialize everything for rendering
        self.reset()
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        mov = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            mov = 1 # accelerate
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            mov = 2 # brake
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            mov = 3 # turn left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            mov = 4 # turn right

        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0
                terminated = False # To restart the loop
    
    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()