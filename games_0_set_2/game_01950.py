
# Generated: 2025-08-27T18:48:15.545830
# Source Brief: brief_01950.md
# Brief Index: 1950

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to draw lines from the pen. Space/Shift to move the pen along the sled's trajectory."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics puzzle where you draw tracks for a sled to reach the finish line. "
        "Inspired by the classic game Line Rider."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.rng = np.random.default_rng()

        # --- Game Constants ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_TERRAIN = (60, 70, 80)
        self.COLOR_LINE = (0, 192, 255)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_START = (0, 255, 0)
        self.COLOR_FINISH = (255, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PEN = (255, 200, 0, 150) # With alpha

        self.GRAVITY = 0.15
        self.FRICTION = 0.98
        self.MAX_STEPS = 2000
        self.DRAW_LINE_LENGTH = 30
        self.PEN_MOVE_SPEED = 20
        self.START_POS = np.array([50.0, 150.0])
        self.FINISH_X = self.SCREEN_WIDTH - 50
        self.STALL_LIMIT = 30 # steps
        
        # Will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sled_pos = None
        self.sled_vel = None
        self.lines = None
        self.terrain_lines = None
        self.pen_pos = None
        self.particles = None
        self.max_distance_reached = 0
        self.stalled_steps = 0
        self.terrain_roughness = 0.0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.sled_pos = self.START_POS.copy()
        self.sled_vel = np.array([1.0, 0.0])
        self.pen_pos = self.sled_pos + np.array([20.0, 20.0])
        
        self.lines = []
        self.particles = []
        
        self.max_distance_reached = self.sled_pos[0]
        self.stalled_steps = 0
        
        self.terrain_roughness = 0.1
        self.terrain_lines = self._generate_terrain(self.terrain_roughness)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Handle Actions
        self._handle_actions(movement, space_held, shift_held)

        # Update Physics
        self._update_physics()
        
        # Update Particles
        self._update_particles()

        # Calculate Reward & Check Termination
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self.sled_pos[0] >= self.FINISH_X:
                reward += 100 # Goal-oriented reward
            else:
                reward -= 10 # Crash penalty

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_actions(self, movement, space_held, shift_held):
        # Priority: Pen movement > Drawing
        if space_held or shift_held:
            sled_vel_norm = np.linalg.norm(self.sled_vel)
            if sled_vel_norm > 0.1:
                direction = self.sled_vel / sled_vel_norm
                if space_held:
                    # 'Move pen forward'
                    self.pen_pos += direction * self.PEN_MOVE_SPEED
                elif shift_held:
                    # 'Move pen backward'
                    self.pen_pos -= direction * self.PEN_MOVE_SPEED
        elif movement != 0:
            # 'Draw line'
            start_pos = self.pen_pos.copy()
            end_pos = self.pen_pos.copy()
            if movement == 1: # Up
                end_pos[1] -= self.DRAW_LINE_LENGTH
            elif movement == 2: # Down
                end_pos[1] += self.DRAW_LINE_LENGTH
            elif movement == 3: # Left
                end_pos[0] -= self.DRAW_LINE_LENGTH
            elif movement == 4: # Right
                end_pos[0] += self.DRAW_LINE_LENGTH
            
            # Add line and move pen to the end
            self.lines.append((start_pos, end_pos))
            self.pen_pos = end_pos

        # Clamp pen position to screen
        self.pen_pos[0] = np.clip(self.pen_pos[0], 0, self.SCREEN_WIDTH)
        self.pen_pos[1] = np.clip(self.pen_pos[1], 0, self.SCREEN_HEIGHT)

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        
        # Predict next position
        start_point = self.sled_pos
        end_point = self.sled_pos + self.sled_vel

        # Collision detection
        min_t = 1.0
        collision_info = None
        all_lines = self.terrain_lines + self.lines

        for line in all_lines:
            p1, p2 = line
            t, u, intersection_point = self._get_line_intersection(start_point, end_point, p1, p2)
            if t is not None and 0 <= t < min_t and 0 <= u <= 1:
                min_t = t
                collision_info = {
                    "line": line,
                    "point": intersection_point,
                }
        
        # Handle collision or move
        if collision_info:
            # Move to collision point
            self.sled_pos = collision_info["point"]

            # Calculate reflection
            line_vec = collision_info["line"][1] - collision_info["line"][0]
            line_normal = np.array([-line_vec[1], line_vec[0]])
            line_normal = line_normal / (np.linalg.norm(line_normal) + 1e-6)

            # Ensure normal points "upwards" relative to sled velocity
            if np.dot(self.sled_vel, line_normal) > 0:
                line_normal *= -1

            # Reflect velocity
            dot_product = np.dot(self.sled_vel, line_normal)
            self.sled_vel -= 2 * dot_product * line_normal
            
            # Apply friction
            self.sled_vel *= self.FRICTION
            # 'Sound: sled scraping on line'
        else:
            # No collision, move normally
            self.sled_pos = end_point

    def _get_line_intersection(self, p1, p2, p3, p4):
        v1 = p2 - p1
        v2 = p4 - p3
        
        A = np.array([v1, -v2]).T
        b = p3 - p1
        
        try:
            x = np.linalg.solve(A, b)
            t, u = x[0], x[1]
            intersection_point = p1 + t * v1
            return t, u, intersection_point
        except np.linalg.LinAlgError:
            return None, None, None

    def _update_particles(self):
        # Add new particle
        if np.linalg.norm(self.sled_vel) > 1.0:
            p_vel = self.rng.random(2) * 2 - 1
            self.particles.append({
                "pos": self.sled_pos.copy(),
                "vel": p_vel,
                "life": self.rng.integers(20, 40)
            })

        # Update existing particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for horizontal progress
        progress = self.sled_pos[0] - self.max_distance_reached
        if progress > 0:
            reward += 0.1 * progress # Reward for moving right
            
            # Event-based reward for new max distance
            if self.sled_pos[0] > self.max_distance_reached:
                reward += 5
                self.max_distance_reached = self.sled_pos[0]
        
        return reward

    def _check_termination(self):
        # Win condition
        if self.sled_pos[0] >= self.FINISH_X:
            return True
        
        # Crash conditions
        if not (0 <= self.sled_pos[0] <= self.SCREEN_WIDTH and 0 <= self.sled_pos[1] <= self.SCREEN_HEIGHT):
            return True
        
        # Stall condition
        if np.linalg.norm(self.sled_vel) < 0.1:
            self.stalled_steps += 1
        else:
            self.stalled_steps = 0
        if self.stalled_steps >= self.STALL_LIMIT:
            return True
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            return True
        
        return False

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.sled_pos[0],
            "max_distance": self.max_distance_reached,
        }

    def _render_game(self):
        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_POS[0], 0), (self.START_POS[0], self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.SCREEN_HEIGHT), 2)

        # Draw terrain
        if len(self.terrain_lines) > 0:
            points = [line[0] for line in self.terrain_lines] + [self.terrain_lines[-1][1]]
            pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, points, 2)
        
        # Draw drawn lines
        for line in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, line[0], line[1], 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40.0))
            color = (self.COLOR_SLED[0], self.COLOR_SLED[1], self.COLOR_SLED[2], alpha)
            radius = int(p["life"] / 10)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

        # Draw sled
        sled_size = 6
        sled_rect = pygame.Rect(
            int(self.sled_pos[0] - sled_size / 2),
            int(self.sled_pos[1] - sled_size / 2),
            sled_size, sled_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect)

        # Draw pen cursor
        pen_size = 5
        x, y = int(self.pen_pos[0]), int(self.pen_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, pen_size, self.COLOR_PEN)
        pygame.draw.line(self.screen, self.COLOR_PEN, (x - pen_size, y), (x + pen_size, y))
        pygame.draw.line(self.screen, self.COLOR_PEN, (x, y - pen_size), (x, y + pen_size))

    def _render_ui(self):
        dist_text = f"Distance: {int(self.max_distance_reached)} / {int(self.FINISH_X)}"
        text_surface = self.font.render(dist_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

    def _generate_terrain(self, roughness):
        # Update difficulty based on steps
        current_roughness = roughness + 0.05 * (self.steps // 500)
        
        points = [(0, self.SCREEN_HEIGHT * 0.6)]
        num_segments = 10
        segment_width = self.SCREEN_WIDTH / num_segments

        for i in range(1, num_segments + 1):
            last_y = points[-1][1]
            y_change = self.rng.uniform(-self.SCREEN_HEIGHT * current_roughness, self.SCREEN_HEIGHT * current_roughness)
            new_y = np.clip(last_y + y_change, self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT * 0.9)
            points.append((i * segment_width, new_y))

        lines = []
        for i in range(len(points) - 1):
            lines.append((np.array(points[i]), np.array(points[i+1])))
        return lines
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a window to display the game
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Game loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Manual Controls ---
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement for drawing
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Pen movement
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability
        
    env.close()