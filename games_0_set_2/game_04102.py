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

    user_guide = (
        "Controls: Use arrow keys to adjust bounce angle. ↑/↓ for large changes, ←/→ for fine tuning."
    )

    game_description = (
        "Navigate a ball through an isometric tunnel maze. Set your bounce angle before hitting a wall to guide the ball to the glowing green exit. You have a time limit and a limited number of wall hits."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 20 * self.FPS  # 20 seconds
        self.MAX_WALL_HITS = 5
        
        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 60)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALL = (100, 110, 130)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_EXIT_GLOW = (150, 255, 200)
        self.COLOR_PARTICLE = (255, 80, 80)
        self.COLOR_INDICATOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HITS_TEXT = (255, 100, 100)

        # --- Game State (Persistent across resets) ---
        self.current_level = 1
        self.max_level = 10
        self._np_random = None

        # --- Game State (Reset every episode) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "WIN", "TIMEOUT", "CRASHED"
        
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_radius = 8
        self.ball_speed = 3.5

        self.walls = []
        self.exit_poly_world = []
        self.exit_poly_screen = []
        self.exit_center_world = pygame.Vector2(0,0)
        
        self.wall_hits = 0
        self.bounce_angle_offset = 0.0  # Player-controlled angle in degrees
        self.particles = []
        self.prev_dist_to_exit = 0.0
        
        self.iso_tile_width = 32
        self.iso_tile_height = 16
        self.world_offset = pygame.Vector2(self.WIDTH / 2, 50)

        # --- Initialize state variables ---
        # self.reset() is called by the wrapper, no need to call it here.

    def _world_to_screen(self, pos):
        x, y = pos
        screen_x = self.world_offset.x + (x - y) * self.iso_tile_width / 2
        screen_y = self.world_offset.y + (x + y) * self.iso_tile_height / 2
        return pygame.Vector2(int(screen_x), int(screen_y))

    def _generate_tunnel(self):
        self.walls = []
        num_corners = min(20, 2 + self.current_level)
        tunnel_width = 4.0
        segment_length_min = 5
        segment_length_max = 10

        path = [pygame.Vector2(0, 0)]
        
        # FIX: Use indexing to select a pygame.Vector2, as np.random.choice can convert it to an ndarray
        initial_directions = [pygame.Vector2(0, 1), pygame.Vector2(1, 0)]
        direction = initial_directions[self._np_random.integers(len(initial_directions))]
        
        turn_angles = [-90, 90]
        for i in range(num_corners):
            length = self._np_random.uniform(segment_length_min, segment_length_max)
            path.append(path[-1] + direction * length)
            # Turn 90 degrees left or right
            angle_to_rotate = turn_angles[self._np_random.integers(len(turn_angles))]
            direction = direction.rotate(angle_to_rotate)
        
        # Final straight segment
        length = self._np_random.uniform(segment_length_min, segment_length_max)
        path.append(path[-1] + direction * length)

        # Create wall segments from path
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            seg_dir = (p2 - p1).normalize()
            normal = seg_dir.rotate(90)
            
            self.walls.append(((p1 + normal * tunnel_width / 2), (p2 + normal * tunnel_width / 2)))
            self.walls.append(((p1 - normal * tunnel_width / 2), (p2 - normal * tunnel_width / 2)))
        
        # Define start and exit
        start_dir = (path[1] - path[0]).normalize()
        self.ball_pos = path[0] + start_dir * 1.5
        self.ball_vel = start_dir * self.ball_speed

        exit_center = path[-1]
        exit_dir = (path[-1] - path[-2]).normalize()
        exit_normal = exit_dir.rotate(90)
        
        self.exit_center_world = exit_center
        w = tunnel_width / 2
        self.exit_poly_world = [
            exit_center - exit_dir * 0.5 - exit_normal * w,
            exit_center - exit_dir * 0.5 + exit_normal * w,
            exit_center + exit_dir * 0.5 + exit_normal * w,
            exit_center + exit_dir * 0.5 - exit_normal * w,
        ]
        self.exit_poly_screen = [self._world_to_screen(p) for p in self.exit_poly_world]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed=seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        if hasattr(self, 'game_outcome') and self.game_outcome == "WIN":
             self.current_level = min(self.max_level, self.current_level + 1)
        # If loss, level stays the same. For a full game restart, one would re-init the class.

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.wall_hits = 0
        self.bounce_angle_offset = 0.0
        self.particles.clear()
        
        self._generate_tunnel()
        
        self.prev_dist_to_exit = self.ball_pos.distance_to(self.exit_center_world)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If game is over, just return the final state
            reward = 0
            return self._get_observation(), reward, True, False, self._get_info()

        # --- 1. Process Action ---
        movement = action[0]
        angle_change = 0
        if movement == 1: angle_change = -15.0 # up = sharp left bounce
        elif movement == 2: angle_change = 15.0 # down = sharp right bounce
        elif movement == 3: angle_change = -7.5  # left = slight left bounce
        elif movement == 4: angle_change = 7.5   # right = slight right bounce
        self.bounce_angle_offset = np.clip(self.bounce_angle_offset + angle_change, -80, 80)
        
        # --- 2. Update Game Logic ---
        self._update_ball()
        self._update_particles()
        self.steps += 1
        
        # --- 3. Calculate Reward & Check Termination ---
        reward = 0
        terminated = False
        
        # Distance-based reward
        current_dist_to_exit = self.ball_pos.distance_to(self.exit_center_world)
        if current_dist_to_exit < self.prev_dist_to_exit:
            reward += 0.01 # Small positive reward for progress
        else:
            reward -= 0.01 # Small negative reward for moving away
        self.prev_dist_to_exit = current_dist_to_exit

        # Event-based rewards and termination checks
        if self.game_over:
            terminated = True
            if self.game_outcome == "WIN":
                reward += 100
            elif self.game_outcome == "CRASHED":
                reward -= 100
            elif self.game_outcome == "TIMEOUT":
                reward -= 100
        
        # Wall hit penalty is applied in _update_ball
        # Add to cumulative score
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            self.game_outcome = "TIMEOUT"
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Per brief, truncated is handled by termination
            self._get_info()
        )

    def _update_ball(self):
        next_pos = self.ball_pos + self.ball_vel

        # --- Collision Detection ---
        for wall_start, wall_end in self.walls:
            # Simple line-circle collision
            line_vec = wall_end - wall_start
            if line_vec.length() == 0: continue
            
            point_vec = next_pos - wall_start
            line_len_sq = line_vec.length_squared()
            
            t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
            closest_point = wall_start + t * line_vec
            
            dist_sq = next_pos.distance_squared_to(closest_point)

            if dist_sq < self.ball_radius**2 / (self.iso_tile_width/2)**2: # Scale radius to world coords
                # --- Collision Response ---
                self.wall_hits += 1
                self.score -= 1 # Immediate penalty for wall hit
                
                # Create particles
                for _ in range(self._np_random.integers(5, 10)):
                    self.particles.append(Particle(self._world_to_screen(self.ball_pos), self._np_random))

                # Bounce logic
                wall_normal = line_vec.rotate(90).normalize()
                if wall_normal.dot(self.ball_vel) > 0:
                    wall_normal = -wall_normal

                perfect_reflection = self.ball_vel.reflect(wall_normal)
                
                # Apply player-controlled rotation
                self.ball_vel = perfect_reflection.rotate(self.bounce_angle_offset)
                self.ball_vel.scale_to_length(self.ball_speed)

                # Move ball slightly out of wall to prevent sticking
                next_pos = self.ball_pos + self.ball_vel 
                
                if self.wall_hits >= self.MAX_WALL_HITS:
                    self.game_over = True
                    self.game_outcome = "CRASHED"
                break
        
        self.ball_pos = next_pos

        # --- Win Condition ---
        if self.exit_poly_screen[0].distance_to(self._world_to_screen(self.ball_pos)) < 20:
             if self._point_in_poly(self.ball_pos, self.exit_poly_world):
                self.game_over = True
                self.game_outcome = "WIN"
        
        # --- Lose Condition (Timeout) ---
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_outcome = "TIMEOUT"
            
    def _point_in_poly(self, point, poly):
        # Ray casting algorithm
        x, y = point
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()

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
            "level": self.current_level,
            "wall_hits": self.wall_hits,
        }

    def _render_game(self):
        # Draw exit area
        pygame.gfxdraw.filled_polygon(self.screen, self.exit_poly_screen, self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, self.exit_poly_screen, self.COLOR_EXIT)

        # Draw walls
        for start, end in self.walls:
            p1 = self._world_to_screen(start)
            p2 = self._world_to_screen(end)
            pygame.draw.line(self.screen, self.COLOR_WALL, p1, p2, 3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self.COLOR_PARTICLE)

        # Draw ball
        ball_screen_pos = self._world_to_screen(self.ball_pos)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.ball_radius+2, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.ball_radius+2, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.ball_radius, self.COLOR_BALL)

        # Draw bounce angle indicator
        if not self.game_over:
            indicator_len = 30
            # Indicator is relative to current velocity
            indicator_angle = self.ball_vel.angle_to(pygame.Vector2(1, 0)) + self.bounce_angle_offset
            indicator_end = ball_screen_pos + pygame.Vector2(indicator_len, 0).rotate(-indicator_angle)
            pygame.draw.aaline(self.screen, self.COLOR_INDICATOR, ball_screen_pos, indicator_end, 2)
    
    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Wall Hits
        hits_text_str = f"HITS: {self.wall_hits}/{self.MAX_WALL_HITS}"
        hits_text = self.font_ui.render(hits_text_str, True, self.COLOR_HITS_TEXT)
        self.screen.blit(hits_text, (self.WIDTH - hits_text.get_width() - 10, 10))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = ""
            color = (255, 255, 255)
            if self.game_outcome == "WIN":
                msg = f"LEVEL {self.current_level} COMPLETE!"
                color = self.COLOR_EXIT
            elif self.game_outcome == "CRASHED":
                msg = "TOO MANY HITS"
                color = self.COLOR_HITS_TEXT
            elif self.game_outcome == "TIMEOUT":
                msg = "TIME OUT"
                color = (200, 200, 255)
            
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()


class Particle:
    def __init__(self, pos, np_random):
        self.pos = pygame.Vector2(pos)
        angle = np_random.uniform(0, 360)
        speed = np_random.uniform(1, 4)
        self.vel = pygame.Vector2(speed, 0).rotate(angle)
        self.lifespan = np_random.uniform(10, 20) # frames
        self.radius = np_random.uniform(2, 4)

    def is_alive(self):
        return self.lifespan > 0

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius -= 0.1
    
    def draw(self, surface, color):
        if self.is_alive() and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 15))))
            temp_color = (*color, alpha)
            # Using a simple filled circle as gfxdraw with alpha is complex
            temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, temp_color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (self.pos.x - self.radius, self.pos.y - self.radius))

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Tunnel Maze")
    clock = pygame.time.Clock()
    running = True
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Display the frame ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            # Pause for 2 seconds on game over before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()