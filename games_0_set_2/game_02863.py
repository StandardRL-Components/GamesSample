
# Generated: 2025-08-28T06:15:39.361271
# Source Brief: brief_02863.md
# Brief Index: 2863

        
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

    # Short, user-facing control string
    user_guide = (
        "Use arrows to move the drawing cursor. Press Space to add a track point, and Shift to remove the last one."
    )

    # Short, user-facing description of the game
    game_description = (
        "Draw a path for your sledder to ride across a dangerous, hilly landscape. Reach the finish line as fast as you can!"
    )

    # Frames auto-advance for smooth physics simulation
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
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
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_BG_MOUNTAIN = (35, 45, 55)
        self.COLOR_TERRAIN = (100, 110, 120)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_TRACK = (0, 192, 255)
        self.COLOR_CURSOR = (0, 192, 255, 150)
        self.COLOR_START = (0, 255, 128)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_OBSTACLE = (200, 100, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SPARK = (255, 220, 180)
        
        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.25
        self.FRICTION = 0.995
        self.RIDER_RADIUS = 8
        self.FINISH_X = 4000
        self.CURSOR_SPEED = 10
        self.MAX_TRACK_POINTS = 50
        self.MAX_TRACK_SEGMENT_LENGTH = 150
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_angle = 0.0
        self.rider_on_ground = False
        self.terrain_points = []
        self.player_track_points = []
        self.obstacles = []
        self.particles = []
        self.camera_x = 0.0
        self.drawing_cursor_pos = pygame.Vector2(0, 0)
        self.checkpoints_reached = 0
        self.last_reward = 0.0
        self.bg_mountains = []

        self.reset()
        # self.validate_implementation() # Uncomment to run validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.checkpoints_reached = 0
        
        # Generate world
        self._generate_world()

        # Reset rider
        self.rider_pos = pygame.Vector2(100, self.terrain_points[0][1] - 50)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_angle = 0.0
        self.rider_on_ground = False

        # Reset drawing
        self.player_track_points = [self.rider_pos.copy()]
        self.drawing_cursor_pos = self.rider_pos + pygame.Vector2(100, 0)

        # Reset camera and particles
        self.camera_x = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_physics()
        self._update_particles()
        self._update_camera()

        self.steps += 1
        
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_world(self):
        self.terrain_points = []
        self.obstacles = []
        self.bg_mountains = []
        
        difficulty_level = self.steps // 500
        max_slope = 2 + difficulty_level * 2
        num_obstacles = 1 + difficulty_level

        # Generate terrain
        y = self.HEIGHT * 0.75
        for x in range(-self.WIDTH, self.FINISH_X + self.WIDTH, 50):
            self.terrain_points.append((x, y))
            y += self.np_random.uniform(-max_slope, max_slope)
            y = np.clip(y, self.HEIGHT * 0.4, self.HEIGHT * 0.9)

        # Generate obstacles
        for _ in range(num_obstacles):
            # Place obstacles on the ground in the middle of the course
            segment_index = self.np_random.integers(10, len(self.terrain_points) - 10)
            p1 = pygame.Vector2(self.terrain_points[segment_index])
            size = self.np_random.integers(20, 40)
            obstacle_rect = pygame.Rect(p1.x - size/2, p1.y - size, size, size)
            self.obstacles.append(obstacle_rect)
        
        # Generate background mountains
        for _ in range(20):
            x = self.np_random.uniform(-self.WIDTH, self.FINISH_X + self.WIDTH)
            y = self.HEIGHT
            w = self.np_random.uniform(200, 600)
            h = self.np_random.uniform(100, 300)
            self.bg_mountains.append((x, y, w, h))

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.drawing_cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.drawing_cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.drawing_cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.drawing_cursor_pos.x += self.CURSOR_SPEED
        
        # Clamp cursor to be near the rider and within screen
        self.drawing_cursor_pos.x = np.clip(self.drawing_cursor_pos.x, self.rider_pos.x - 200, self.rider_pos.x + 200)
        self.drawing_cursor_pos.y = np.clip(self.drawing_cursor_pos.y, 0, self.HEIGHT)

        # Add track point
        if space_pressed and len(self.player_track_points) < self.MAX_TRACK_POINTS:
            last_point = self.player_track_points[-1]
            if last_point.distance_to(self.drawing_cursor_pos) > 10 and last_point.distance_to(self.drawing_cursor_pos) < self.MAX_TRACK_SEGMENT_LENGTH:
                self.player_track_points.append(self.drawing_cursor_pos.copy())
                # Sound: "click"

        # Remove track point
        if shift_pressed and len(self.player_track_points) > 1:
            self.player_track_points.pop()
            # Sound: "undo"

    def _update_physics(self):
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        self.rider_vel *= self.FRICTION
        
        # Predict next position
        next_pos = self.rider_pos + self.rider_vel
        
        # Collision detection with all lines (terrain + player track)
        all_lines = []
        for i in range(len(self.terrain_points) - 1):
            all_lines.append((pygame.Vector2(self.terrain_points[i]), pygame.Vector2(self.terrain_points[i+1])))
        for i in range(len(self.player_track_points) - 1):
            all_lines.append((self.player_track_points[i], self.player_track_points[i+1]))

        collided = False
        target_angle = self.rider_angle
        
        for p1, p2 in all_lines:
            # Broad-phase check
            if not (min(p1.x, p2.x) - self.RIDER_RADIUS < next_pos.x < max(p1.x, p2.x) + self.RIDER_RADIUS):
                continue
            
            # Closest point on line segment to rider
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            t = ((next_pos - p1).dot(line_vec)) / line_vec.length_squared()
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec
            
            dist_vec = next_pos - closest_point
            if dist_vec.length() < self.RIDER_RADIUS:
                # Collision occurred
                collided = True
                
                # Position correction
                overlap = self.RIDER_RADIUS - dist_vec.length()
                self.rider_pos -= dist_vec.normalize() * overlap
                next_pos = self.rider_pos + self.rider_vel

                # Velocity correction
                normal = dist_vec.normalize()
                
                # Only react to surfaces we are falling onto
                if normal.dot(self.rider_vel) < 0:
                    # Rider is landing on this surface
                    if not self.rider_on_ground:
                        # Hard landing effect
                        for _ in range(10):
                            self._create_particle(self.rider_pos, self.COLOR_SPARK, 3, 30)
                        # Sound: "thump"
                    
                    # Project velocity onto the surface tangent
                    tangent = pygame.Vector2(-normal.y, normal.x)
                    speed = self.rider_vel.length()
                    self.rider_vel = tangent * tangent.dot(self.rider_vel)
                    
                    # Add a small bounce/push-off force
                    self.rider_vel += normal * 1.0 
                    
                    target_angle = math.degrees(math.atan2(line_vec.y, line_vec.x))
                    break # Process one collision per frame
        
        self.rider_on_ground = collided
        self.rider_pos += self.rider_vel
        
        # Smoothly interpolate rider angle
        angle_diff = (target_angle - self.rider_angle + 180) % 360 - 180
        self.rider_angle += angle_diff * 0.2
        
        # Create snow kick-up particles
        if self.rider_on_ground and self.rider_vel.length() > 1:
            if self.np_random.random() < 0.5:
                self._create_particle(self.rider_pos, self.COLOR_RIDER, 1, 20, is_snow=True)

        # Obstacle collision
        rider_rect = pygame.Rect(self.rider_pos.x - self.RIDER_RADIUS, self.rider_pos.y - self.RIDER_RADIUS, self.RIDER_RADIUS*2, self.RIDER_RADIUS*2)
        for obs in self.obstacles:
            if rider_rect.colliderect(obs):
                self.game_over = True
                self.last_reward = -10 # Crash penalty
                # Sound: "crash_explosion"
                for _ in range(30):
                    self._create_particle(self.rider_pos, self.COLOR_OBSTACLE, 4, 60)
                return

    def _create_particle(self, pos, color, size, lifetime, is_snow=False):
        if is_snow:
            vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1, 0)) - self.rider_vel * 0.1
        else:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel,
            'lifetime': lifetime,
            'max_lifetime': lifetime,
            'color': color,
            'size': self.np_random.uniform(size-1, size+1)
        })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.05 # Particle gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _update_camera(self):
        target_cam_x = self.rider_pos.x - self.WIDTH * 0.25
        self.camera_x += (target_cam_x - self.camera_x) * 0.1

    def _calculate_reward(self):
        if self.game_over:
            return self.last_reward

        reward = 0
        
        # Reward for forward movement
        if self.rider_vel.x > 0:
            reward += 0.1 * min(self.rider_vel.x / 5.0, 1.0)
        
        # Penalty for slow speed
        if self.rider_vel.length() < 1.0:
            reward -= 0.01

        # Penalty for being unstable (not flat)
        unstable_factor = abs(self.rider_angle) / 90.0
        if self.rider_on_ground and unstable_factor > 0.5:
             reward -= 0.1 * unstable_factor

        # Checkpoint reward
        current_checkpoint = int(self.rider_pos.x // 1000)
        if current_checkpoint > self.checkpoints_reached:
            self.checkpoints_reached = current_checkpoint
            reward += 5.0
        
        # Goal reward
        if self.rider_pos.x >= self.FINISH_X:
            reward += 100
            self.game_over = True
            self.last_reward = reward

        self.last_reward = reward
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Rider fell out of the world
        if self.rider_pos.y > self.HEIGHT + 100 or self.rider_pos.x < self.camera_x - 50:
            self.game_over = True
            self.last_reward = -10 # Crash penalty
            return True
            
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render background ---
        for x, y, w, h in self.bg_mountains:
            screen_x = x - self.camera_x * 0.5
            if -w < screen_x < self.WIDTH:
                points = [(screen_x, y), (screen_x + w/2, y-h), (screen_x + w, y)]
                pygame.draw.polygon(self.screen, self.COLOR_BG_MOUNTAIN, points)

        # --- Render world elements (camera-adjusted) ---
        
        # Terrain
        terrain_screen_points = [(p[0] - self.camera_x, p[1]) for p in self.terrain_points]
        pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, terrain_screen_points, 2)
        
        # Player track
        if len(self.player_track_points) > 1:
            track_screen_points = [(p.x - self.camera_x, p.y) for p in self.player_track_points]
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, track_screen_points, 3)

        # Start/Finish Lines
        start_pos = 100 - self.camera_x
        finish_pos = self.FINISH_X - self.camera_x
        pygame.draw.line(self.screen, self.COLOR_START, (start_pos, 0), (start_pos, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_pos, 0), (finish_pos, self.HEIGHT), 3)

        # Obstacles
        for obs in self.obstacles:
            screen_obs = obs.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_obs)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = p['color']
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            size = int(p['size'] * (p['lifetime'] / p['max_lifetime']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color + (alpha,))

        # Rider
        rider_screen_pos = (int(self.rider_pos.x - self.camera_x), int(self.rider_pos.y))
        # Rider body (sled)
        rotated_surf = pygame.Surface((self.RIDER_RADIUS * 3, self.RIDER_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(rotated_surf, self.COLOR_RIDER, (0, self.RIDER_RADIUS//2, self.RIDER_RADIUS*3, self.RIDER_RADIUS))
        rotated_surf = pygame.transform.rotate(rotated_surf, -self.rider_angle)
        rect = rotated_surf.get_rect(center=rider_screen_pos)
        self.screen.blit(rotated_surf, rect)
        # Rider head (simple circle)
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1] - self.RIDER_RADIUS, self.RIDER_RADIUS // 2, self.COLOR_RIDER)


        # --- Render foreground elements (not camera-adjusted) ---
        
        # Drawing cursor
        cursor_screen_pos = (int(self.drawing_cursor_pos.x - self.camera_x), int(self.drawing_cursor_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 8, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 8, self.COLOR_CURSOR)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

        # Speed
        speed_kmh = self.rider_vel.length() * 10
        speed_text = self.font_small.render(f"Speed: {speed_kmh:.0f} km/h", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, 10))
        
        # Distance to finish
        dist = max(0, self.FINISH_X - self.rider_pos.x)
        dist_text = self.font_small.render(f"Dist: {dist/100:.1f}m", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (self.WIDTH - dist_text.get_width() - 10, 35))

        # Game Over message
        if self.game_over:
            message = "FINISH!" if self.rider_pos.x >= self.FINISH_X else "CRASHED"
            end_text = self.font_large.render(message, True, self.COLOR_FINISH if message == "CRASHED" else self.COLOR_START)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
            "distance_to_finish": max(0, self.FINISH_X - self.rider_pos.x)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Sled Rider")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            # Optional: auto-reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    env.close()