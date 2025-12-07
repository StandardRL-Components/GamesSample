
# Generated: 2025-08-28T00:31:28.969538
# Source Brief: brief_03816.md
# Brief Index: 3816

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class TrackSegment:
    """Represents a single piece of the sled track."""
    def __init__(self, start_pos, start_angle, seg_type, length, angle_change, resolution=10):
        self.start_pos = np.array(start_pos, dtype=float)
        self.seg_type = seg_type
        self.points = [self.start_pos.copy()]
        self.length = length
        self.color = (50, 150, 255) # Bright Blue

        current_pos = self.start_pos.copy()
        current_angle = start_angle
        step_len = length / resolution
        
        for i in range(resolution):
            if seg_type == 'curve':
                current_angle += angle_change / resolution
            
            delta_x = math.cos(current_angle) * step_len
            delta_y = math.sin(current_angle) * step_len
            current_pos += [delta_x, delta_y]
            self.points.append(current_pos.copy())
            
        self.end_pos = self.points[-1]
        self.end_angle = current_angle
        self.x_min = min(p[0] for p in self.points)
        self.x_max = max(p[0] for p in self.points)

    def draw(self, surface):
        """Draws the track segment with anti-aliasing."""
        if len(self.points) > 1:
            point_list = [(int(p[0]), int(p[1])) for p in self.points]
            pygame.draw.aalines(surface, self.color, False, point_list, 2)

    def get_surface_at(self, x_pos):
        """Finds the track height and angle at a given x-position."""
        if not (self.x_min <= x_pos <= self.x_max):
            return None

        # Find the two points on the segment that bracket the x_pos
        p1, p2 = None, None
        for i in range(len(self.points) - 1):
            if self.points[i][0] <= x_pos <= self.points[i+1][0] or \
               self.points[i+1][0] <= x_pos <= self.points[i][0]:
                p1, p2 = self.points[i], self.points[i+1]
                break
        
        if p1 is None:
            return None

        # Handle vertical lines
        if abs(p1[0] - p2[0]) < 1e-6:
            return p1[1] if p1[0] > p2[0] else p2[1], math.pi / 2

        # Linear interpolation
        t = (x_pos - p1[0]) / (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        
        return y, angle

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.pos = [x, y]
        self.vel = [random.uniform(-1, 1), random.uniform(-2, 0)]
        self.color = color
        self.lifespan = random.randint(15, 30)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1  # Gravity
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            size = max(1, int(3 * (self.lifespan / 30)))
            rect = pygame.Rect(int(self.pos[0]), int(self.pos[1]), size, size)
            
            # Create a temporary surface for alpha blending
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*self.color, alpha))
            surface.blit(s, rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑/↓ to draw curves, ←/→ to tilt. Space for long straights, Shift to wait. Build a track to the finish!"
    )

    game_description = (
        "A physics-based puzzle game. Draw a track for your sled to reach the finish line before time runs out. Choose your segments wisely to build the fastest path."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 35, 45)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_START = (0, 200, 100)
        self.COLOR_FINISH = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)

        # Game constants
        self.GRAVITY = 0.15
        self.FRICTION = 0.995
        self.MAX_STEPS = 1000
        self.FINISH_X = self.WIDTH - 50
        self.CHECKPOINT_X = self.WIDTH / 2
        
        # Sled state
        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.sled_angle = 0.0
        self.on_track = True

        # Track state
        self.track_segments = []
        self.builder_pos = np.array([0.0, 0.0])
        self.builder_angle = 0.0

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.checkpoint_reached = False
        self.termination_reason = ""
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.checkpoint_reached = False
        self.termination_reason = ""
        
        self.sled_pos = np.array([50.0, self.HEIGHT / 2.0])
        self.sled_vel = np.array([1.0, 0.0])
        self.sled_angle = 0.0
        self.on_track = True

        self.track_segments = []
        start_platform = TrackSegment([20, self.HEIGHT / 2], 0, 'straight', 60, 0)
        self.track_segments.append(start_platform)
        self.builder_pos = start_platform.end_pos
        self.builder_angle = start_platform.end_angle

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        prev_sled_x = self.sled_pos[0]

        self._handle_action(action)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1

        terminated = self._check_termination()
        reward = self._calculate_reward(prev_sled_x, terminated)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if shift_held:
            # Player chooses to wait, no new track
            # Sound: a soft 'tick' or 'click'
            return

        angle_tilt = 0.0
        if movement == 3: angle_tilt = -math.radians(15) # Angle Up
        if movement == 4: angle_tilt = math.radians(15)  # Angle Down

        base_angle = self.builder_angle + angle_tilt
        
        seg_type = 'straight'
        length = 30 if not space_held else 60
        angle_change = 0.0

        if movement == 1: # Curve Up
            seg_type = 'curve'
            angle_change = -math.radians(45)
        elif movement == 2: # Curve Down
            seg_type = 'curve'
            angle_change = math.radians(45)
        
        new_segment = TrackSegment(self.builder_pos, base_angle, seg_type, length, angle_change)
        
        # Prevent track from going too far out of bounds
        if new_segment.end_pos[0] < self.WIDTH and new_segment.end_pos[1] > 0 and new_segment.end_pos[1] < self.HEIGHT:
            self.track_segments.append(new_segment)
            self.builder_pos = new_segment.end_pos
            self.builder_angle = new_segment.end_angle
            # Sound: 'zip' or 'swoosh' for track placement

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        surface_info = None
        for segment in reversed(self.track_segments):
            surface_info = segment.get_surface_at(self.sled_pos[0])
            if surface_info:
                break
        
        was_on_track = self.on_track
        self.on_track = False

        if surface_info:
            track_y, track_angle = surface_info
            # Check if sled is on or just above the track
            if self.sled_pos[1] >= track_y - 5 and self.sled_vel[1] >= 0:
                self.on_track = True
                
                # Snap to track and align angle
                self.sled_pos[1] = track_y
                self.sled_angle = track_angle

                # Project velocity onto track tangent
                tangent = np.array([math.cos(track_angle), math.sin(track_angle)])
                speed = np.dot(self.sled_vel, tangent)
                self.sled_vel = tangent * speed

                # Apply friction
                self.sled_vel *= self.FRICTION

                # Create landing particles
                if not was_on_track:
                    # Sound: 'thud' on landing
                    for _ in range(5):
                        self.particles.append(Particle(self.sled_pos[0], self.sled_pos[1], (200, 200, 200)))

        # Update position
        self.sled_pos += self.sled_vel
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        if self.game_over:
            return True
            
        # Win condition
        if self.sled_pos[0] >= self.FINISH_X:
            self.game_over = True
            self.termination_reason = "FINISHED!"
            # Sound: 'cheer' or 'fanfare'
            return True
        
        # Crash condition
        x, y = self.sled_pos
        if not (0 < x < self.WIDTH and -50 < y < self.HEIGHT + 50):
            self.game_over = True
            self.termination_reason = "CRASHED!"
            # Sound: 'crash' or 'explosion'
            return True

        # Timeout condition
        if self.time_left <= 0:
            self.game_over = True
            self.termination_reason = "TIME OUT!"
            # Sound: 'buzzer'
            return True
            
        return False

    def _calculate_reward(self, prev_sled_x, terminated):
        reward = 0.0

        # Movement reward
        progress = self.sled_pos[0] - prev_sled_x
        reward += progress * 0.1

        # Checkpoint reward
        if not self.checkpoint_reached and self.sled_pos[0] >= self.CHECKPOINT_X:
            reward += 5.0
            self.checkpoint_reached = True

        if terminated:
            if self.termination_reason == "FINISHED!":
                reward += 100.0
            elif self.termination_reason == "CRASHED!":
                reward -= 50.0
            elif self.termination_reason == "TIME OUT!":
                reward -= 10.0
        
        return reward

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

        # Draw start/finish/checkpoint lines
        pygame.draw.line(self.screen, self.COLOR_START, (20, 0), (20, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.HEIGHT), 2)
        pygame.draw.line(self.screen, (100, 100, 150), (self.CHECKPOINT_X, 0), (self.CHECKPOINT_X, self.HEIGHT), 1)

        # Draw track
        for segment in self.track_segments:
            segment.draw(self.screen)
        
        # Draw builder cursor
        if not self.game_over:
            cursor_end_x = self.builder_pos[0] + 20 * math.cos(self.builder_angle)
            cursor_end_y = self.builder_pos[1] + 20 * math.sin(self.builder_angle)
            pygame.draw.aaline(self.screen, (255, 255, 0, 100), self.builder_pos, (cursor_end_x, cursor_end_y))
            pygame.gfxdraw.filled_circle(self.screen, int(self.builder_pos[0]), int(self.builder_pos[1]), 3, (255, 255, 0, 100))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw sled
        sled_size = 8
        points = [
            (-sled_size, -sled_size/2), (sled_size, -sled_size/2),
            (sled_size, sled_size/2), (-sled_size, sled_size/2)
        ]
        
        cos_a, sin_a = math.cos(self.sled_angle), math.sin(self.sled_angle)
        
        rotated_points = []
        for x, y in points:
            rx = x * cos_a - y * sin_a + self.sled_pos[0]
            ry = x * sin_a + y * cos_a + self.sled_pos[1]
            rotated_points.append((int(rx), int(ry)))
        
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED)

        # Draw speed glow
        speed = np.linalg.norm(self.sled_vel)
        if speed > 1:
            glow_radius = int(min(20, speed * 2))
            glow_alpha = int(min(100, speed * 15))
            glow_color = (*(255, 255, 100), glow_alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(self.sled_pos[0]), int(self.sled_pos[1]), glow_radius, glow_color)

    def _render_ui(self):
        # Time
        time_text = self.font_ui.render(f"TIME: {self.time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Speed
        speed = np.linalg.norm(self.sled_vel)
        speed_text = self.font_ui.render(f"SPEED: {speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        
        if self.game_over:
            msg_text = self.font_msg.render(self.termination_reason, True, self.COLOR_FINISH if "CRASH" in self.termination_reason else self.COLOR_START)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "sled_pos": self.sled_pos.tolist(),
            "sled_vel": self.sled_vel.tolist(),
            "checkpoint_reached": self.checkpoint_reached
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'dummy' as needed

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Rider")
    
    terminated = False
    total_reward = 0
    
    # Mapping keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: (1, 0, 0),
        pygame.K_DOWN: (2, 0, 0),
        pygame.K_LEFT: (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 0, 1),
        pygame.K_RSHIFT: (0, 0, 1),
    }

    print("--- Controls ---")
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # This logic prioritizes shift > space > movement
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action = np.array([0, 0, 1])
            elif keys[pygame.K_SPACE]:
                action = np.array([0, 1, 0])
            else:
                move_action = 0
                if keys[pygame.K_UP]: move_action = 1
                elif keys[pygame.K_DOWN]: move_action = 2
                elif keys[pygame.K_LEFT]: move_action = 3
                elif keys[pygame.K_RIGHT]: move_action = 4
                action = np.array([move_action, 0, 0])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(60) # Control the speed of the manual play

    env.close()