import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:09:31.672506
# Source Brief: brief_00916.md
# Brief Index: 916
# """import gymnasium as gym
class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Ramp Roll'.

    The player controls the tilt of a procedurally generated ramp to guide a
    ball, collecting gems and avoiding falling off. The game features a
    simple 3D perspective, particle effects, and a focus on game feel.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (3=tilt left, 4=tilt right, others=no-op)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Observation Space:** Box shape=(400, 640, 3), dtype=uint8
    - A rendered RGB image of the game state.

    **Rewards:**
    - +10 for each gem collected.
    - +0.01 for each step the ball stays on the ramp.
    - -5 for falling off the ramp.
    - +100 terminal reward for reaching the score goal.
    - -100 terminal reward for running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a rolling ball down a procedurally generated ramp. Tilt the ramp to collect gems and avoid falling off before time runs out."
    user_guide = "Use the ← and → arrow keys to tilt the ramp and guide the ball."
    auto_advance = True

    # --- Constants ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 60
    MAX_STEPS = 3600 # 60 seconds at 60 FPS

    # Colors
    COLOR_BG = (15, 18, 42)
    COLOR_BALL = (0, 150, 255)
    COLOR_BALL_GLOW = (0, 100, 200)
    COLOR_RAMP = (100, 105, 120)
    COLOR_RAMP_SIDE = (70, 75, 90)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (200, 150, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (30, 30, 30)
    COLOR_LIVES = (255, 80, 80)

    # Game Mechanics
    INITIAL_LIVES = 3
    SCORE_GOAL = 100
    
    # Physics
    BALL_FALL_SPEED = 2.5  # How fast the world scrolls down
    TILT_ACCELERATION = 0.15
    FRICTION = 0.95
    MAX_VELOCITY_X = 5.0

    # Rewards
    REWARD_GEM = 10.0
    REWARD_FALL = -5.0
    REWARD_SURVIVE = 0.01
    REWARD_WIN = 100.0
    REWARD_TIMEOUT = -100.0
    
    # 3D Camera
    CAMERA_Y = 150 # Height above the ramp
    CAMERA_Z_OFFSET = -200 # Distance in front of the ball
    FOCAL_LENGTH = 300

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.time_remaining = 0
        self.ball_pos = None
        self.ball_vel_x = 0
        self.camera_z = 0
        self.ramp_segments = []
        self.gems = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.time_remaining = self.MAX_STEPS
        
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, 500.0]) # [x, z] in world coords
        self.ball_vel_x = 0.0
        self.camera_z = self.ball_pos[1] + self.CAMERA_Z_OFFSET

        self.particles.clear()
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = 0
        terminated = False
        truncated = False

        # --- Update Game Logic ---
        # 1. Apply player input
        tilt = 0
        if movement == 3: # Left
            tilt = -1
        elif movement == 4: # Right
            tilt = 1
        
        self.ball_vel_x += tilt * self.TILT_ACCELERATION
        
        # 2. Apply physics
        self.ball_vel_x *= self.FRICTION
        self.ball_vel_x = np.clip(self.ball_vel_x, -self.MAX_VELOCITY_X, self.MAX_VELOCITY_X)
        self.ball_pos[0] += self.ball_vel_x
        self.ball_pos[1] += self.BALL_FALL_SPEED

        # 3. Collision detection and state updates
        current_segment = self._get_current_ramp_segment()
        if current_segment:
            half_width = current_segment['width'] / 2
            if abs(self.ball_pos[0] - current_segment['center_x']) <= half_width:
                reward += self.REWARD_SURVIVE
            else:
                # Fell off
                reward += self.REWARD_FALL
                self.lives -= 1
                # SFX: Fall sound
                self._create_particles(self.ball_pos, (200,200,200), 20, is_fall=True)
                # Reset ball position
                self.ball_pos[0] = current_segment['center_x']
                self.ball_vel_x = 0
        else: # Off the end of the generated ramp (should not happen in normal play)
            self.lives = 0

        # Gem collection
        collected_indices = []
        for i, gem in enumerate(self.gems):
            dist = np.linalg.norm(self.ball_pos - gem['pos'])
            if dist < gem['radius'] + 10: # 10 is ball radius
                collected_indices.append(i)
                self.score += 1
                reward += self.REWARD_GEM
                # SFX: Gem collect sound
                self._create_particles(gem['pos'], self.COLOR_GEM, 15)
        
        # Remove collected gems safely
        for i in sorted(collected_indices, reverse=True):
            del self.gems[i]

        # 4. Update game state
        self.steps += 1
        self.time_remaining -= 1
        self._update_particles()
        self._prune_and_spawn_entities()

        # 5. Check for termination
        if self.score >= self.SCORE_GOAL:
            terminated = True
            reward += self.REWARD_WIN
        elif self.lives <= 0:
            terminated = True
        elif self.time_remaining <= 0:
            terminated = True
            reward += self.REWARD_TIMEOUT
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Smooth camera follow
        self.camera_z = self.camera_z * 0.9 + (self.ball_pos[1] + self.CAMERA_Z_OFFSET) * 0.1
        
        self.screen.fill(self.COLOR_BG)
        self._render_ramp()
        self._render_gems()
        self._render_particles()
        self._render_ball()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
            "time_remaining": self.time_remaining
        }

    def _generate_level(self):
        self.ramp_segments.clear()
        self.gems.clear()
        
        # Generate ramp
        z = 0
        center_x = self.SCREEN_WIDTH / 2
        
        # Use multiple sine waves for more interesting curves
        freq1, amp1 = self.np_random.uniform(150, 250), self.np_random.uniform(100, 200)
        freq2, amp2 = self.np_random.uniform(300, 400), self.np_random.uniform(50, 100)
        phase1, phase2 = self.np_random.uniform(0, 2*math.pi), self.np_random.uniform(0, 2*math.pi)

        num_segments = int(self.MAX_STEPS * self.BALL_FALL_SPEED) + self.SCREEN_HEIGHT
        
        for i in range(num_segments):
            z += 1
            offset = math.sin(z / freq1 + phase1) * amp1 + math.sin(z / freq2 + phase2) * amp2
            center_x = self.SCREEN_WIDTH / 2 + offset
            width = 150 - 40 * abs(math.cos(z/freq1 + phase1)) # Varying width
            self.ramp_segments.append({'z': z, 'center_x': center_x, 'width': width})

        # Generate gems
        for i in range(100): # Sprinkle gems along the ramp
            segment_index = self.np_random.integers(200, len(self.ramp_segments))
            segment = self.ramp_segments[segment_index]
            gem_x = segment['center_x'] + self.np_random.uniform(-segment['width']/2.5, segment['width']/2.5)
            gem_z = segment['z']
            self.gems.append({'pos': np.array([gem_x, gem_z]), 'radius': 15})

    def _prune_and_spawn_entities(self):
        # Remove gems that are far behind the camera
        self.gems = [g for g in self.gems if g['pos'][1] > self.camera_z - 100]

    def _project(self, x, y, z):
        """Projects 3D world coordinates to 2D screen coordinates."""
        dz = z - self.camera_z
        if dz <= 0: return None # Behind camera
        
        scale = self.FOCAL_LENGTH / dz
        screen_x = self.SCREEN_WIDTH / 2 + (x - self.SCREEN_WIDTH / 2) * scale
        screen_y = self.SCREEN_HEIGHT / 2 + (y - self.CAMERA_Y) * scale
        
        return int(screen_x), int(screen_y), scale

    def _render_ramp(self):
        # Find first visible segment
        start_index = 0
        for i, seg in enumerate(self.ramp_segments):
            if seg['z'] > self.camera_z:
                start_index = max(0, i - 1)
                break
        
        for i in range(start_index, len(self.ramp_segments) - 1):
            s1 = self.ramp_segments[i]
            s2 = self.ramp_segments[i+1]
            
            p1 = self._project(s1['center_x'] - s1['width']/2, 0, s1['z'])
            p2 = self._project(s1['center_x'] + s1['width']/2, 0, s1['z'])
            p3 = self._project(s2['center_x'] + s2['width']/2, 0, s2['z'])
            p4 = self._project(s2['center_x'] - s2['width']/2, 0, s2['z'])
            
            # Stop if segment is off-screen
            if p3 is None or p3[1] > self.SCREEN_HEIGHT + 20: break
            if p1 is None: continue

            points = [p1[:2], p2[:2], p3[:2], p4[:2]]
            
            # Draw side polygons for 3D effect
            side_height = 20 * p1[2] # Make side appear constant height
            side_color = self.COLOR_RAMP_SIDE
            
            # Left side
            left_side_points = [p1[:2], p4[:2], (p4[0], p4[1] + side_height), (p1[0], p1[1] + side_height)]
            pygame.gfxdraw.aapolygon(self.screen, left_side_points, side_color)
            pygame.gfxdraw.filled_polygon(self.screen, left_side_points, side_color)
            
            # Right side
            right_side_points = [p2[:2], p3[:2], (p3[0], p3[1] + side_height), (p2[0], p2[1] + side_height)]
            pygame.gfxdraw.aapolygon(self.screen, right_side_points, side_color)
            pygame.gfxdraw.filled_polygon(self.screen, right_side_points, side_color)

            # Draw top surface
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_RAMP)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_RAMP)

    def _render_ball(self):
        proj = self._project(self.ball_pos[0], 0, self.ball_pos[1])
        if proj:
            sx, sy, scale = proj
            radius = max(2, int(10 * scale))
            
            # Shadow
            shadow_proj = self._project(self.ball_pos[0], -radius, self.ball_pos[1])
            if shadow_proj:
                pygame.gfxdraw.filled_ellipse(self.screen, shadow_proj[0], shadow_proj[1], int(radius*1.1), int(radius*0.5), (0,0,0,50))

            # Glow
            glow_radius = int(radius * 1.8)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, (*self.COLOR_BALL_GLOW, 100))
            
            # Ball
            pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_BALL)

    def _render_gems(self):
        for gem in self.gems:
            proj = self._project(gem['pos'][0], 15, gem['pos'][1]) # y=15 to hover
            if proj:
                sx, sy, scale = proj
                if -50 < sx < self.SCREEN_WIDTH + 50 and -50 < sy < self.SCREEN_HEIGHT + 50:
                    radius = max(2, int(gem['radius'] * scale))
                    
                    # Pulsing glow
                    glow_alpha = 100 + 50 * math.sin(self.steps * 0.1)
                    glow_radius = int(radius * (1.5 + 0.2 * math.sin(self.steps * 0.1)))
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, (*self.COLOR_GEM_GLOW, int(glow_alpha)))
                    
                    # Gem body
                    pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_GEM)
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_GEM)

    def _render_particles(self):
        for p in self.particles:
            proj = self._project(p['pos'][0], p['pos_y'], p['pos'][1])
            if proj:
                sx, sy, scale = proj
                radius = max(1, int(p['size'] * p['life'] * scale))
                alpha = int(255 * p['life'])
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= p['decay']
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _create_particles(self, pos, color, count, is_fall=False):
        for _ in range(count):
            if is_fall:
                vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(0, 1)]) * 2
                pos_y = 0
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed * 0.5 + self.BALL_FALL_SPEED])
                pos_y = 15
            
            self.particles.append({
                'pos': pos.copy(),
                'pos_y': pos_y,
                'vel': vel,
                'life': 1.0,
                'decay': self.np_random.uniform(0.01, 0.03),
                'color': color,
                'size': self.np_random.uniform(3, 7)
            })

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0]+2, pos[1]+2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        # Score
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10))
        
        # Lives
        lives_text = "LIVES: "
        draw_text(lives_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 70 + i*25, 22, 8, self.COLOR_LIVES)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 70 + i*25, 22, 8, (255,255,255))
        
        # Timer
        time_str = f"{int(self.time_remaining / self.GAME_FPS):02d}"
        draw_text(time_str, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH/2 - 20, 5))

    def _get_current_ramp_segment(self):
        ball_z = int(self.ball_pos[1])
        if 0 <= ball_z < len(self.ramp_segments):
            return self.ramp_segments[ball_z]
        return None

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This part requires a display. It will not run in a headless environment.
    # To run, comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at the top.
    
    env = GameEnv()
    obs, info = env.reset()
    
    try:
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Ramp Roll - Manual Play")
        running = True
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        running = False # Cannot run manual play loop

    terminated = False
    total_reward = 0
    
    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0

        action = [0, 0, 0] # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_r]:
            terminated = True # Force reset

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Blit the observation from the env's internal surface to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.GAME_FPS)

    env.close()