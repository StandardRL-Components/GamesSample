
# Generated: 2025-08-28T03:26:45.964889
# Source Brief: brief_02013.md
# Brief Index: 2013

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a 2D physics-based puzzle game.
    The player draws lines to guide a sled down a procedurally generated slope
    to reach a finish line as quickly as possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to aim, hold Space to draw a line. "
        "Hold Shift+Space for a shorter, precision line. "
        "Guide the sled to the red finish line."
    )

    game_description = (
        "A minimalist physics puzzler. Draw lines on the screen to create a path for your sled. "
        "Navigate the procedurally generated terrain and reach the finish line before time runs out. "
        "Every win makes the next slope steeper!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 35, 60)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_SLED_INNER = (200, 200, 255)
        self.COLOR_LINE = (0, 0, 0)
        self.COLOR_TERRAIN = (150, 160, 180)
        self.COLOR_START = (0, 255, 100)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)

        # Physics & Game Constants
        self.GRAVITY = pygame.math.Vector2(0, 0.25)
        self.SLED_RADIUS = 8
        self.AIR_DRAG = 0.998
        self.FRICTION = 0.1
        self.BOUNCE = 0.1
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.STUCK_LIMIT = 10 # steps
        self.MIN_SPEED_FOR_STUCK_CHECK = 0.1

        # Game State
        self.difficulty_level = 1.0 # Controls slope angle
        self.np_random = None
        
        # Initialize state variables
        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.lines = []
        self.terrain = []
        self.particles = []
        self.start_line_y = 0
        self.finish_line_rect = pygame.Rect(0,0,0,0)
        self.steps = 0
        self.score = 0
        self.time_left_steps = 0
        self.stuck_counter = 0
        self.game_over = False
        self.last_dist_to_finish = 0.0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left_steps = self.MAX_STEPS
        self.stuck_counter = 0

        self.lines = []
        self.particles = []
        
        self._generate_terrain()
        
        self.last_dist_to_finish = self._get_dist_to_finish()

        return self._get_observation(), self._get_info()

    def _generate_terrain(self):
        self.terrain = []
        points = []
        
        start_x, start_y = 50, 100
        points.append(pygame.math.Vector2(0, start_y))
        points.append(pygame.math.Vector2(start_x, start_y))
        
        self.start_line_y = start_y
        self.sled_pos = pygame.math.Vector2(start_x, start_y - self.SLED_RADIUS * 2)
        self.sled_vel = pygame.math.Vector2(0, 0)

        curr_x, curr_y = start_x, start_y
        
        base_slope = 0.2 * self.difficulty_level
        
        while curr_x < self.WIDTH - 100:
            segment_len = self.np_random.integers(40, 80)
            slope = base_slope + self.np_random.uniform(-0.1, 0.3)
            
            next_x = curr_x + segment_len
            next_y = curr_y + segment_len * slope
            
            points.append(pygame.math.Vector2(next_x, next_y))
            curr_x, curr_y = next_x, next_y

        points.append(pygame.math.Vector2(self.WIDTH, curr_y))
        
        for i in range(len(points) - 1):
            self.terrain.append((points[i], points[i+1]))
            
        finish_y = max(p.y for p in points) + 20
        self.finish_line_rect = pygame.Rect(0, finish_y, self.WIDTH, 5)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        line_drawn_length = self._handle_input(action)
        self._update_physics()
        self._update_game_state()
        
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated, line_drawn_length)
        self.score += reward
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        line_drawn_length = 0

        if space_held and movement != 0:
            length = 5 if shift_held else 20
            direction_map = {
                1: pygame.math.Vector2(0, -1), # Up
                2: pygame.math.Vector2(0, 1),  # Down
                3: pygame.math.Vector2(-1, 0), # Left
                4: pygame.math.Vector2(1, 0),  # Right
            }
            direction = direction_map[movement]
            
            start_point = self.sled_pos.copy()
            end_point = self.sled_pos + direction * length
            
            self.lines.append((start_point, end_point))
            # sound: "line_draw_swoosh.wav"
            
            # Limit total number of lines to prevent performance issues
            if len(self.lines) > 50:
                self.lines.pop(0)
            
            line_drawn_length = length
        
        return line_drawn_length

    def _update_physics(self):
        # Apply gravity and air drag
        self.sled_vel += self.GRAVITY
        self.sled_vel *= self.AIR_DRAG

        # Collision detection and response
        collided = False
        all_lines = self.terrain + self.lines
        
        for p1, p2 in all_lines:
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            line_normal = line_vec.rotate(90).normalize()
            
            # Ensure normal points "upwards" relative to sled
            if line_normal.dot(self.sled_pos - p1) < 0:
                line_normal = -line_normal
                
            # Closest point on infinite line
            proj = (self.sled_pos - p1).dot(line_vec.normalize())
            
            # Check if closest point is on segment
            if 0 <= proj <= line_vec.length():
                closest_point = p1 + proj * line_vec.normalize()
            else:
                closest_point = p1 if proj < 0 else p2
            
            dist_vec = self.sled_pos - closest_point
            
            if dist_vec.length() < self.SLED_RADIUS:
                collided = True
                
                # Resolve penetration
                penetration = self.SLED_RADIUS - dist_vec.length()
                self.sled_pos += line_normal * penetration
                
                # Collision response
                v_normal = self.sled_vel.dot(line_normal)
                if v_normal < 0: # Moving towards the line
                    v_tangent_vec = self.sled_vel - v_normal * line_normal
                    
                    self.sled_vel = v_tangent_vec * (1 - self.FRICTION) - (v_normal * self.BOUNCE * line_normal)

                    # Create particles on collision/slide
                    if self.sled_vel.length() > 1:
                        # sound: "sled_slide_snow.wav"
                        for _ in range(2):
                            p_vel = -self.sled_vel.normalize().rotate(random.uniform(-30, 30)) * random.uniform(0.5, 2)
                            self.particles.append([self.sled_pos.copy(), p_vel, random.uniform(2, 4)])

        # Update position
        self.sled_pos += self.sled_vel

    def _update_game_state(self):
        self.time_left_steps -= 1

        # Update particles
        self.particles = [
            [p[0] + p[1], p[1] * 0.9, p[2] - 0.1]
            for p in self.particles if p[2] > 0
        ]
        
        # Stuck check
        if self.sled_vel.length() < self.MIN_SPEED_FOR_STUCK_CHECK:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def _check_termination(self):
        win = self.finish_line_rect.collidepoint(self.sled_pos.x, self.sled_pos.y)
        loss_bounds = not (0 < self.sled_pos.x < self.WIDTH and -50 < self.sled_pos.y < self.HEIGHT + 50)
        loss_timeout = self.time_left_steps <= 0
        loss_stuck = self.stuck_counter >= self.STUCK_LIMIT
        
        terminated = win or loss_bounds or loss_timeout or loss_stuck
        if terminated:
            self.game_over = True
            if win:
                # sound: "win_jingle.wav"
                self.difficulty_level += 0.25 # Increase difficulty for next game
            else:
                # sound: "lose_sad_trombone.wav"
                pass
        return terminated

    def _calculate_reward(self, terminated, line_drawn_length):
        if terminated:
            if self.finish_line_rect.collidepoint(self.sled_pos.x, self.sled_pos.y):
                return 100.0  # Win
            return -100.0 # Loss
        
        current_dist = self._get_dist_to_finish()
        
        # Reward for getting closer to the finish line
        progress_reward = (self.last_dist_to_finish - current_dist) * 0.5
        
        # Penalty for drawing lines (encourages efficiency)
        line_penalty = line_drawn_length * 0.05
        
        self.last_dist_to_finish = current_dist
        
        reward = progress_reward - line_penalty
        return reward

    def _get_dist_to_finish(self):
        # Use horizontal distance to the right edge as a proxy for progress
        return self.WIDTH - self.sled_pos.x

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (0, self.start_line_y), (100, self.start_line_y), 5)
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_line_rect)

        # Draw terrain
        for p1, p2 in self.terrain:
            pygame.draw.line(self.screen, self.COLOR_TERRAIN, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 3)
            
        # Draw player lines
        for p1, p2 in self.lines:
            pygame.draw.line(self.screen, self.COLOR_LINE, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 4)
            
        # Draw particles
        for p in self.particles:
            pos, _, size = p
            if size > 0:
                alpha = int(max(0, min(255, (size / 4.0) * 255)))
                color = (*self.COLOR_SLED, alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(size), int(size)), int(size))
                self.screen.blit(temp_surf, (int(pos.x - size), int(pos.y - size)))

        # Draw sled
        sled_x, sled_y = int(self.sled_pos.x), int(self.sled_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.filled_circle(self.screen, sled_x, sled_y, self.SLED_RADIUS - 3, self.COLOR_SLED_INNER)
        
        # Draw direction indicator
        if self.sled_vel.length_squared() > 0.1:
            direction_vec = self.sled_vel.normalize()
            p1 = self.sled_pos + direction_vec * (self.SLED_RADIUS-2)
            p2 = self.sled_pos - direction_vec * 2
            pygame.draw.line(self.screen, self.COLOR_BG, (int(p2.x), int(p2.y)), (int(p1.x), int(p1.y)), 2)

    def _render_ui(self):
        time_left_sec = max(0, self.time_left_steps // self.FPS)
        timer_text = f"TIME: {time_left_sec:02d}"
        text_surface = self.font.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.time_left_steps // self.FPS),
            "difficulty": self.difficulty_level,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game window setup
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    print(env.user_guide)
    
    while not done:
        # --- Player Input ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
                print("--- Game Reset ---")

        # --- Game Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered game screen
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Info: {info}")
            
    env.close()