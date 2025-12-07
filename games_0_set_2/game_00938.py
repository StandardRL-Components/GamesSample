import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to accelerate. Avoid blue obstacles and reach the green finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down procedural racer. Navigate a Tron-like track, dodge moving obstacles, and race against the clock to the finish."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_OBSTACLE_GLOW = (100, 200, 255, 50)
    COLOR_TRACK = (220, 220, 220)
    COLOR_FINISH_LINE = (50, 255, 50)
    COLOR_PARTICLE = (255, 220, 50)
    COLOR_TEXT = (240, 240, 240)

    # Game parameters
    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    TRACK_LENGTH = 8000
    TRACK_WIDTH = 400
    TRACK_WAYPOINTS = 50
    NUM_OBSTACLES = 150

    # Physics
    ACCELERATION = 0.4
    MAX_SPEED = 8.0
    DRAG = 0.97
    PLAYER_SIZE = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 40)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.waypoints = []
        self.obstacles = []
        self.particles = []
        self.finish_line_y = None
        self.obstacle_speed_multiplier = 1.0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        # self.reset() is called after seed is set in super().reset()
        
        # Run validation check
        # self.validate_implementation()

    def _generate_track(self):
        self.waypoints = []
        # The center of the track must be within the screen, leaving room for half the track's width on either side.
        low_bound = self.TRACK_WIDTH / 2
        high_bound = self.SCREEN_WIDTH - self.TRACK_WIDTH / 2

        # The original code caused a ValueError because the low argument to uniform() was greater than the high argument.
        # self.TRACK_WIDTH * 1.5 = 600
        # self.SCREEN_WIDTH - self.TRACK_WIDTH * 1.5 = 640 - 600 = 40
        # np.random.uniform(600, 40) is invalid.
        # The corrected bounds [200, 440] are valid.
        start_x = self.np_random.uniform(low_bound, high_bound)
        
        y_points = np.linspace(self.TRACK_LENGTH, 100, self.TRACK_WAYPOINTS)
        
        # Generate a smooth path using a random walk for x
        current_x = start_x
        for y in y_points:
            self.waypoints.append(np.array([current_x, y]))
            step = self.np_random.uniform(-150, 150)
            current_x += step
            # The original clip range was also invalid: np.clip(..., 400, 240).
            # We use the same corrected bounds here to ensure the track stays on screen.
            current_x = np.clip(current_x, low_bound, high_bound)
        
        self.finish_line_y = self.waypoints[-1][1]

    def _generate_obstacles(self):
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            # Find a random segment on the track
            if len(self.waypoints) < 2: continue
            segment_index = self.np_random.integers(0, len(self.waypoints) - 1)
            p1 = self.waypoints[segment_index]
            p2 = self.waypoints[segment_index + 1]
            
            # Interpolate a random point along the segment
            interp = self.np_random.random()
            center_pos = p1 + (p2 - p1) * interp
            
            # Place obstacle randomly to the side of the center line
            offset = self.np_random.uniform(30, self.TRACK_WIDTH / 2 - 20)
            angle = self.np_random.choice([-np.pi / 2, np.pi / 2])
            
            pos = center_pos + np.array([offset * np.cos(angle), offset * np.sin(angle)])

            motion_type = self.np_random.choice(['static', 'h_osc', 'v_osc', 'circle'])
            size = self.np_random.uniform(15, 30)
            
            self.obstacles.append({
                'base_pos': pos,
                'pos': pos.copy(),
                'size': size,
                'motion_type': motion_type,
                'motion_phase': self.np_random.uniform(0, 2 * np.pi),
                'motion_range': self.np_random.uniform(20, 60),
            })
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_track()
        self._generate_obstacles()

        self.player_pos = self.waypoints[0].copy() + np.array([0, 100]) # Start just behind the first waypoint
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = -np.pi / 2  # Start facing up

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.obstacle_speed_multiplier = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # Calculate distance to finish line before moving
        dist_before = np.linalg.norm(self.player_pos - self.waypoints[-1])
        
        # Player physics
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] -= self.ACCELERATION # Up
        if movement == 2: accel[1] += self.ACCELERATION # Down
        if movement == 3: accel[0] -= self.ACCELERATION # Left
        if movement == 4: accel[0] += self.ACCELERATION # Right
        
        self.player_vel += accel
        self.player_vel *= self.DRAG
        
        # Cap speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.MAX_SPEED
        
        self.player_pos += self.player_vel
        
        # Update player angle for rendering
        if np.linalg.norm(self.player_vel) > 0.1:
            self.player_angle = math.atan2(self.player_vel[1], self.player_vel[0]) + np.pi/2

        # Update obstacles and difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_speed_multiplier += 0.05

        self._update_obstacles()
        self._update_particles()
        
        # Calculate distance to finish line after moving
        dist_after = np.linalg.norm(self.player_pos - self.waypoints[-1])

        # --- Reward Calculation ---
        reward = 0
        
        # Reward for progress towards finish line
        progress_reward = dist_before - dist_after
        reward += progress_reward * 0.1 # Scaled reward for moving closer
        
        # --- Termination Checks ---
        terminated = False
        
        # 1. Collision with obstacle
        if self._check_collisions():
            reward = -10.0
            self._create_explosion(self.player_pos)
            terminated = True
            # sfx: player_explosion
            
        # 2. Crossed finish line
        if self.player_pos[1] < self.finish_line_y:
            reward = 100.0
            terminated = True
            # sfx: win_jingle
        
        # 3. Timeout
        if self.steps >= self.MAX_STEPS:
            reward = -5.0 # Penalty for running out of time
            terminated = True
            # sfx: timeout_buzzer

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_obstacles(self):
        time = self.steps * 0.05 * self.obstacle_speed_multiplier
        for obs in self.obstacles:
            offset = np.array([0.0, 0.0])
            if obs['motion_type'] == 'h_osc':
                offset[0] = obs['motion_range'] * np.sin(time + obs['motion_phase'])
            elif obs['motion_type'] == 'v_osc':
                offset[1] = obs['motion_range'] * np.sin(time + obs['motion_phase'])
            elif obs['motion_type'] == 'circle':
                offset[0] = obs['motion_range'] * np.cos(time + obs['motion_phase'])
                offset[1] = obs['motion_range'] * np.sin(time + obs['motion_phase'])
            obs['pos'] = obs['base_pos'] + offset

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE/2, 
            self.player_pos[1] - self.PLAYER_SIZE/2, 
            self.PLAYER_SIZE, 
            self.PLAYER_SIZE
        )
        for obs in self.obstacles:
            # Broad-phase check
            if abs(self.player_pos[0] - obs['pos'][0]) > obs['size'] + self.PLAYER_SIZE:
                continue
            if abs(self.player_pos[1] - obs['pos'][1]) > obs['size'] + self.PLAYER_SIZE:
                continue
                
            obs_rect = pygame.Rect(
                obs['pos'][0] - obs['size']/2, 
                obs['pos'][1] - obs['size']/2, 
                obs['size'], 
                obs['size']
            )
            if player_rect.colliderect(obs_rect):
                return True
        return False

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera is centered on the player
        cam_offset = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]) - self.player_pos

        # Render track boundaries
        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i]
            p2 = self.waypoints[i+1]
            
            # Simple culling
            if (p1[1] + cam_offset[1] > self.SCREEN_HEIGHT and p2[1] + cam_offset[1] > self.SCREEN_HEIGHT) or \
               (p1[1] + cam_offset[1] < 0 and p2[1] + cam_offset[1] < 0):
                continue

            # Calculate perpendicular vector for track edges
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm == 0: continue
            perp = np.array([-direction[1], direction[0]]) / norm
            
            w = self.TRACK_WIDTH / 2
            
            # Left edge
            p1_l = tuple(map(int, p1 + perp * w + cam_offset))
            p2_l = tuple(map(int, p2 + perp * w + cam_offset))
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1_l, p2_l)

            # Right edge
            p1_r = tuple(map(int, p1 - perp * w + cam_offset))
            p2_r = tuple(map(int, p2 - perp * w + cam_offset))
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1_r, p2_r)

        # Render finish line
        finish_p1 = (self.waypoints[-1] + np.array([-self.TRACK_WIDTH/2, 0]) + cam_offset).astype(int)
        finish_p2 = (self.waypoints[-1] + np.array([self.TRACK_WIDTH/2, 0]) + cam_offset).astype(int)
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, tuple(finish_p1), tuple(finish_p2), 5)
        
        # Render obstacles
        for obs in self.obstacles:
            screen_pos = obs['pos'] + cam_offset
            if -obs['size'] < screen_pos[0] < self.SCREEN_WIDTH + obs['size'] and \
               -obs['size'] < screen_pos[1] < self.SCREEN_HEIGHT + obs['size']:
                
                size = obs['size']
                rect = pygame.Rect(screen_pos[0] - size/2, screen_pos[1] - size/2, size, size)
                
                # Glow effect
                glow_rect = rect.inflate(size * 0.5, size * 0.5)
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW, glow_surf.get_rect(), border_radius=int(size/4))
                self.screen.blit(glow_surf, glow_rect.topleft)

                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=int(size/5))

        # Render particles
        for p in self.particles:
            screen_pos = p['pos'] + cam_offset
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p['size']), color)

        # Render player
        if not (self.game_over and self._check_collisions()): # Don't draw player if they just crashed
            player_screen_pos = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
            
            # Points for the triangle
            p1 = np.array([0, -self.PLAYER_SIZE])
            p2 = np.array([-self.PLAYER_SIZE/1.5, self.PLAYER_SIZE/1.5])
            p3 = np.array([self.PLAYER_SIZE/1.5, self.PLAYER_SIZE/1.5])

            # Rotation matrix
            rot_matrix = np.array([
                [np.cos(self.player_angle), -np.sin(self.player_angle)],
                [np.sin(self.player_angle), np.cos(self.player_angle)]
            ])

            points = [
                player_screen_pos + p1 @ rot_matrix,
                player_screen_pos + p2 @ rot_matrix,
                player_screen_pos + p3 @ rot_matrix,
            ]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            
            # Glow effect
            glow_points = [
                player_screen_pos + (p1*1.8) @ rot_matrix,
                player_screen_pos + (p2*1.8) @ rot_matrix,
                player_screen_pos + (p3*1.8) @ rot_matrix,
            ]
            int_glow_points = [(int(p[0]), int(p[1])) for p in glow_points]
            pygame.gfxdraw.aapolygon(self.screen, int_glow_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, int_glow_points, self.COLOR_PLAYER_GLOW)

            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps) / 30.0 # Assuming 30fps
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
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
    # It will not work in a headless environment
    try:
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Procedural Racer")
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        render_screen = None

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        if render_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Player Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        else: # If headless, just sample actions
            action = env.action_space.sample()

        # --- Step Environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Render ---
        if render_screen:
            # The observation is already a rendered frame
            # We just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {total_reward:.2f}")
            if render_screen:
                pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            terminated = False
            total_reward = 0

        # Control frame rate
        env.clock.tick(30)

    env.close()