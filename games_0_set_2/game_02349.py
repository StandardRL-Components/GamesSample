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

    user_guide = (
        "Controls: Arrows to move the drawing cursor. Space to place a track point. Shift to remove the last point."
    )

    game_description = (
        "Draw a track in real-time for a sled to navigate a procedurally generated canyon. Reach the finish line without crashing."
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TERRAIN = (102, 72, 54)
        self.COLOR_TRACK = (200, 215, 225)
        self.COLOR_SLED = (255, 60, 60)
        self.COLOR_SLED_GLOW = (255, 100, 100)
        self.COLOR_START = (60, 220, 60)
        self.COLOR_FINISH = (220, 60, 60)
        self.COLOR_CURSOR = (255, 255, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (220, 220, 220)

        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.15
        self.CURSOR_SPEED = 8
        self.SLED_SIZE = 8
        self.START_X = 60
        self.FINISH_X = self.WIDTH - 60
        self.FRICTION = 0.99
        self.SLIDE_FRICTION = 0.995
        self.MIN_SPEED_REWARD = 0.5
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.terrain_points = []
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_vel = pygame.Vector2(0, 0)
        self.track_points = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.checkpoints = []
        self.cleared_checkpoints = set()
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed=seed)
        
        # Initialize state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_terrain()
        
        start_y = self._get_terrain_height(self.START_X) - 30
        self.sled_pos = pygame.Vector2(self.START_X, start_y)
        self.sled_vel = pygame.Vector2(0, 0)
        
        # The track must start at the sled's initial position
        self.track_points = [self.sled_pos.copy()]
        
        self.cursor_pos = self.sled_pos.copy() + pygame.Vector2(20, 0)
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []
        
        # Define checkpoints
        num_checkpoints = 3
        self.checkpoints = [
            self.START_X + (i + 1) * (self.FINISH_X - self.START_X) / (num_checkpoints + 1)
            for i in range(num_checkpoints)
        ]
        self.cleared_checkpoints = set()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # 1. Handle player input (action)
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)
        
        # Place track point
        if space_held and not self.last_space_held:
            if len(self.track_points) < 200: # Limit track length
                self.track_points.append(self.cursor_pos.copy())
                reward -= 1 # Cost for placing a track segment
        
        # Remove track point
        if shift_held and not self.last_shift_held:
            if len(self.track_points) > 1: # Can't remove the starting point
                self.track_points.pop()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # 2. Update game logic
        self._update_sled_physics()
        self._update_particles()
        
        # 3. Calculate rewards
        if self.sled_vel.magnitude() > self.MIN_SPEED_REWARD:
            reward += 0.1

        for i, cp_x in enumerate(self.checkpoints):
            if i not in self.cleared_checkpoints and self.sled_pos.x > cp_x:
                reward += 10
                self.cleared_checkpoints.add(i)

        # 4. Check for termination
        terminated = False
        if self.sled_pos.x > self.FINISH_X:
            reward += 100
            terminated = True
        
        terrain_y = self._get_terrain_height(self.sled_pos.x)
        if self.sled_pos.y > terrain_y - self.SLED_SIZE / 2:
            reward -= 10
            terminated = True

        if not (0 < self.sled_pos.x < self.WIDTH and 0 < self.sled_pos.y < self.HEIGHT):
            reward -= 10
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_sled_physics(self):
        # Apply gravity
        self.sled_vel.y += self.GRAVITY
        self.sled_vel *= self.FRICTION

        if len(self.track_points) > 1:
            # Find closest track segment and perform collision
            min_dist_sq = float('inf')
            closest_seg_info = None

            for i in range(len(self.track_points) - 1):
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                
                # Simple bounding box check to cull distant segments
                if not (min(p1.x, p2.x) - 30 < self.sled_pos.x < max(p1.x, p2.x) + 30):
                    continue

                line_vec = p2 - p1
                if line_vec.length_squared() == 0: continue
                
                point_vec = self.sled_pos - p1
                t = point_vec.dot(line_vec) / line_vec.length_squared()
                t = np.clip(t, 0, 1)
                
                closest_point = p1 + t * line_vec
                dist_sq = (self.sled_pos - closest_point).length_squared()

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_seg_info = (p1, p2, closest_point)

            if closest_seg_info and min_dist_sq < (self.SLED_SIZE * 1.5)**2:
                p1, p2, closest_point = closest_seg_info
                
                # Collision response
                dist_vec = self.sled_pos - closest_point
                if dist_vec.length() < self.SLED_SIZE:
                    
                    # FIX: The original code crashed on dist_vec.normalize() if dist_vec had zero length.
                    # We now check for a non-zero length before attempting to normalize and apply bounce physics.
                    # A small epsilon is used for floating point safety.
                    if dist_vec.length_squared() > 1e-9:
                        # Reposition sled to be on the surface
                        self.sled_pos = closest_point + dist_vec.normalize() * self.SLED_SIZE
                        
                        # Calculate surface normal
                        track_normal = (p2 - p1).rotate(90).normalize()
                        if track_normal.dot(dist_vec) < 0:
                            track_normal = -track_normal
                        
                        # Project velocity onto the normal and dampen it (bounce)
                        vel_dot_normal = self.sled_vel.dot(track_normal)
                        if vel_dot_normal > 0:
                            self.sled_vel -= 1.1 * vel_dot_normal * track_normal
                    
                    self.sled_vel *= self.SLIDE_FRICTION

                    # Spawn particles based on turning
                    vel_dir = self.sled_vel.normalize() if self.sled_vel.length() > 0 else pygame.Vector2(0,0)
                    track_dir = (p2-p1).normalize()
                    turn_severity = 1 - abs(vel_dir.dot(track_dir))
                    if self.sled_vel.length() > 2 and turn_severity > 0.1:
                        self._spawn_particles(self.sled_pos, self.sled_vel, int(turn_severity * 5))


        self.sled_pos += self.sled_vel

    def _spawn_particles(self, pos, base_vel, count):
        for _ in range(count):
            vel = -base_vel.normalize() * self.np_random.uniform(0.5, 2)
            vel.rotate_ip(self.np_random.uniform(-30, 30))
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _generate_terrain(self):
        self.terrain_points = []
        y = self.HEIGHT * 0.75
        
        # Use a sum of sine waves for smooth but varied terrain
        freq1, amp1 = self.np_random.uniform(1, 3), self.np_random.uniform(self.HEIGHT * 0.05, self.HEIGHT * 0.15)
        freq2, amp2 = self.np_random.uniform(4, 7), self.np_random.uniform(self.HEIGHT * 0.02, self.HEIGHT * 0.05)
        phase1, phase2 = self.np_random.uniform(0, 2*math.pi), self.np_random.uniform(0, 2*math.pi)

        for i in range(self.WIDTH + 1):
            # Flatten start and end
            flat_factor = 1
            if i < self.START_X + 50:
                flat_factor = max(0, (i - self.START_X) / 50)
            elif i > self.FINISH_X - 50:
                flat_factor = max(0, (self.FINISH_X - i) / 50)

            sine_val = math.sin(i / self.WIDTH * freq1 * 2 * math.pi + phase1) * amp1
            sine_val += math.sin(i / self.WIDTH * freq2 * 2 * math.pi + phase2) * amp2
            
            self.terrain_points.append((i, y + sine_val * flat_factor))

    def _get_terrain_height(self, x):
        x_clamped = int(np.clip(x, 0, self.WIDTH))
        return self.terrain_points[x_clamped][1]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw terrain
        pygame.gfxdraw.filled_polygon(self.screen, self.terrain_points + [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)], self.COLOR_TERRAIN)
        pygame.gfxdraw.aapolygon(self.screen, self.terrain_points + [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)], self.COLOR_TERRAIN)
        
        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_X, 0), (self.START_X, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.HEIGHT), 2)

        # Draw checkpoints
        for cp_x in self.checkpoints:
             pygame.draw.line(self.screen, (100, 100, 150, 100), (cp_x, 0), (cp_x, self.HEIGHT), 1)

        # Draw player-drawn track
        if len(self.track_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, self.track_points, 2)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, p['life'] * 255 / 30)
            size = max(1, p['life'] * 3 / 30)
            color = self.COLOR_PARTICLE
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(size), (*color, int(alpha)))
            
        # Draw sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE * 2, self.SLED_SIZE * 2)
        sled_rect.center = self.sled_pos
        
        # Glow effect
        glow_radius = int(self.SLED_SIZE * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_SLED_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (sled_rect.centerx - glow_radius, sled_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect, border_radius=2)
        
        # Draw cursor
        if not self.game_over:
            c = self.COLOR_CURSOR
            x, y = int(self.cursor_pos.x), int(self.cursor_pos.y)
            pygame.draw.line(self.screen, c, (x - 8, y), (x + 8, y), 1)
            pygame.draw.line(self.screen, c, (x, y - 8), (x, y + 8), 1)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        speed = self.sled_vel.magnitude() * 10
        speed_text = self.font_small.render(f"SPEED: {speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-set the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen_width, screen_height = 960, 600
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Sled Drawer")
    
    terminated = False
    truncated = False
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not terminated and not truncated:
            # Player controls
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display screen
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
        display_screen.blit(scaled_surface, (0, 0))

        # Display game over message
        if terminated or truncated:
            font = pygame.font.SysFont("monospace", 50, bold=True)
            text = font.render("GAME OVER", True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width/2, screen_height/2 - 30))
            display_screen.blit(text, text_rect)

            font_small = pygame.font.SysFont("monospace", 30)
            text_restart = font_small.render("Press 'R' to restart", True, (200, 200, 200))
            text_restart_rect = text_restart.get_rect(center=(screen_width/2, screen_height/2 + 30))
            display_screen.blit(text_restart, text_restart_rect)

        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()