import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:26:42.948668
# Source Brief: brief_02560.md
# Brief Index: 2560
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate a constantly rotating and growing line through a field of obstacles to reach the goal."
    )
    user_guide = (
        "Use the ← and → arrow keys to shrink and grow your line to avoid the red obstacles and touch the green goal."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CENTER = (WIDTH // 2, HEIGHT // 2)
    FPS = 60
    MAX_STEPS = 30 * FPS  # 30 seconds

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 100, 150)
    COLOR_OBSTACLE = (255, 50, 80)
    COLOR_OBSTACLE_GLOW = (150, 20, 40)
    COLOR_GOAL = (50, 255, 100)
    COLOR_GOAL_GLOW = (20, 150, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (255, 200, 100)

    # Game Parameters
    PLAYER_ROTATION_SPEED = math.radians(2.0)  # Radians per frame
    PLAYER_LENGTH_CHANGE_SPEED = 0.15
    PLAYER_MIN_LENGTH = 1.0
    PLAYER_MAX_LENGTH = 25.0
    SPIRAL_THRESHOLD = 12.0
    SCALE = 12.0  # Pixels per game unit

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "WIN", "LOSE_COLLISION", "LOSE_TIME"
        
        self.player_angle = 0.0
        self.player_length = self.PLAYER_MIN_LENGTH
        
        self.obstacles = []
        self.particles = []
        self.reward_length_thresholds = {2: False, 4: False, 6: False, 8: False}

        self.goal_pos = (0, 0)
        self.goal_radius = 0
        
        # Initialize state by calling reset
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        
        self.player_angle = self.np_random.uniform(0, 2 * math.pi)
        self.player_length = self.PLAYER_MIN_LENGTH
        
        self.particles = []
        self.reward_length_thresholds = {k: False for k in self.reward_length_thresholds}

        # --- Procedurally Generate Level ---
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = 0.01 # Small reward for surviving
        
        self._update_player(movement)
        self._update_obstacles()
        self._update_particles()
        
        # --- Check Game Events & Calculate Reward ---
        terminated = False
        
        # Check length reward
        for threshold, awarded in self.reward_length_thresholds.items():
            if not awarded and self.player_length >= threshold:
                reward += 1.0
                self.reward_length_thresholds[threshold] = True
        
        # Check termination conditions
        if self._check_collision():
            # sfx: explosion
            self.game_over = True
            terminated = True
            self.game_outcome = "COLLISION"
            reward = -100.0
            self._create_explosion(self._get_player_tip_pos(), 50)
        elif self._check_goal():
            # sfx: success chime
            self.game_over = True
            terminated = True
            self.game_outcome = "GOAL REACHED"
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            # sfx: timeout buzzer
            self.game_over = True
            terminated = True
            self.game_outcome = "TIME OUT"
            reward = -50.0

        self.score += reward
        
        # The truncated flag is handled by the environment wrapper
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Methods ---
    def _update_player(self, movement_action):
        self.player_angle = (self.player_angle + self.PLAYER_ROTATION_SPEED) % (2 * math.pi)
        
        if movement_action == 3:  # Left: shrink
            self.player_length -= self.PLAYER_LENGTH_CHANGE_SPEED
            # sfx: shrink sound
        elif movement_action == 4:  # Right: grow
            self.player_length += self.PLAYER_LENGTH_CHANGE_SPEED
            # sfx: grow sound
        
        self.player_length = np.clip(self.player_length, self.PLAYER_MIN_LENGTH, self.PLAYER_MAX_LENGTH)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['angle'] = (obs['angle'] + obs['speed']) % (2 * math.pi)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    # --- Event Checking ---
    def _check_collision(self):
        player_points = self._get_player_collision_points()
        for obs in self.obstacles:
            rotated_verts = [self._rotate_point(v, obs['angle'], (0,0)) for v in obs['vertices']]
            scaled_verts = [(self.CENTER[0] + x * self.SCALE, self.CENTER[1] + y * self.SCALE) for x, y in rotated_verts]
            
            for p_point in player_points:
                if self._point_in_polygon(p_point, scaled_verts):
                    return True
        return False

    def _check_goal(self):
        tip_pos = self._get_player_tip_pos()
        dist_sq = (tip_pos[0] - self.goal_pos[0])**2 + (tip_pos[1] - self.goal_pos[1])**2
        return dist_sq < self.goal_radius**2

    # --- Level Generation ---
    def _generate_level(self):
        self.obstacles = []
        min_dist_from_center = 6.0
        max_dist_from_center = (self.HEIGHT / 2 / self.SCALE) - 2.0

        for i in range(7):
            dist = self.np_random.uniform(min_dist_from_center, max_dist_from_center)
            angle = self.np_random.uniform(0, 2 * math.pi)
            
            pos_x = dist * math.cos(angle)
            pos_y = dist * math.sin(angle)
            
            num_verts = self.np_random.integers(3, 6)
            size = self.np_random.uniform(1.5, 2.5)
            
            vertices = []
            for j in range(num_verts):
                v_angle = 2 * math.pi * j / num_verts
                v_x = pos_x + size * math.cos(v_angle)
                v_y = pos_y + size * math.sin(v_angle)
                vertices.append((v_x, v_y))
            
            speed = self.np_random.uniform(-0.03, 0.03)
            if abs(speed) < 0.005: speed = 0.005 * np.sign(speed) if speed != 0 else 0.005

            self.obstacles.append({
                'vertices': vertices,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'speed': speed
            })
        
        # Place goal
        goal_dist = self.np_random.uniform(self.HEIGHT / 4 / self.SCALE, self.HEIGHT / 2 / self.SCALE - 3.0)
        goal_angle = self.np_random.uniform(0, 2 * math.pi)
        self.goal_pos = (
            self.CENTER[0] + goal_dist * self.SCALE * math.cos(goal_angle),
            self.CENTER[1] + goal_dist * self.SCALE * math.sin(goal_angle)
        )
        self.goal_radius = 2.0 * self.SCALE

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_goal()
        self._render_obstacles()
        self._render_player()
        self._render_particles()

    def _render_goal(self):
        pos = (int(self.goal_pos[0]), int(self.goal_pos[1]))
        radius = int(self.goal_radius)
        # Glow
        for i in range(10, 0, -2):
            alpha = 100 - i * 10
            color = (*self.COLOR_GOAL_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, color)
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_GOAL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_GOAL)

    def _render_obstacles(self):
        for obs in self.obstacles:
            rotated_verts = [self._rotate_point(v, obs['angle'], (0,0)) for v in obs['vertices']]
            scaled_verts = [(self.CENTER[0] + x * self.SCALE, self.CENTER[1] + y * self.SCALE) for x,y in rotated_verts]
            
            # Glow effect
            pygame.draw.polygon(self.screen, self.COLOR_OBSTACLE_GLOW, scaled_verts, width=10)
            # Main shape
            pygame.gfxdraw.filled_polygon(self.screen, scaled_verts, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aapolygon(self.screen, scaled_verts, self.COLOR_OBSTACLE)

    def _render_player(self):
        length_px = self.player_length * self.SCALE

        if self.player_length < self.SPIRAL_THRESHOLD:
            # Render as a line
            end_x = self.CENTER[0] + length_px * math.cos(self.player_angle)
            end_y = self.CENTER[1] + length_px * math.sin(self.player_angle)
            
            # Glow
            for i in range(5, 0, -1):
                pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, self.CENTER, (end_x, end_y), width=i * 3 + 1)
            # Main line
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER, self.CENTER, (end_x, end_y))
            pygame.draw.line(self.screen, self.COLOR_PLAYER, self.CENTER, (end_x, end_y), width=3)
        else:
            # Render as a spiral
            points = []
            final_theta = (self.player_length - self.SPIRAL_THRESHOLD) * 1.5 + 2 * math.pi
            a = length_px / final_theta
            
            for i in range(100):
                theta = i / 99 * final_theta + self.player_angle
                r = a * (i / 99 * final_theta)
                x = self.CENTER[0] + r * math.cos(theta)
                y = self.CENTER[1] + r * math.sin(theta)
                points.append((x, y))
            
            if len(points) > 1:
                # Glow
                pygame.draw.lines(self.screen, self.COLOR_PLAYER_GLOW, False, points, width=10)
                # Main spiral
                pygame.draw.aalines(self.screen, self.COLOR_PLAYER, False, points)
                pygame.draw.lines(self.screen, self.COLOR_PLAYER, False, points, width=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['size'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score:.0f}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surface, (10, 10))
        
        # Game Over Message
        if self.game_over:
            msg = self.game_outcome
            color = self.COLOR_GOAL if msg == "GOAL REACHED" else self.COLOR_OBSTACLE
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            text_surface = self.font_game_over.render(msg, True, color)
            text_rect = text_surface.get_rect(center=self.CENTER)
            self.screen.blit(text_surface, text_rect)

    # --- Helper & Utility Methods ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_player_tip_pos(self):
        length_px = self.player_length * self.SCALE
        if self.player_length < self.SPIRAL_THRESHOLD:
            end_x = self.CENTER[0] + length_px * math.cos(self.player_angle)
            end_y = self.CENTER[1] + length_px * math.sin(self.player_angle)
            return (end_x, end_y)
        else:
            final_theta = (self.player_length - self.SPIRAL_THRESHOLD) * 1.5 + 2 * math.pi
            a = length_px / final_theta
            theta = final_theta + self.player_angle
            r = a * final_theta
            x = self.CENTER[0] + r * math.cos(theta)
            y = self.CENTER[1] + r * math.sin(theta)
            return (x, y)

    def _get_player_collision_points(self):
        points = []
        length_px = self.player_length * self.SCALE

        if self.player_length < self.SPIRAL_THRESHOLD:
            # Sample points along the line
            for i in range(1, 11):
                p = i / 10.0
                x = self.CENTER[0] + p * length_px * math.cos(self.player_angle)
                y = self.CENTER[1] + p * length_px * math.sin(self.player_angle)
                points.append((x, y))
        else:
            # Sample points along the spiral
            final_theta = (self.player_length - self.SPIRAL_THRESHOLD) * 1.5 + 2 * math.pi
            a = length_px / final_theta
            
            num_samples = int(self.player_length) # More samples for longer spiral
            for i in range(1, num_samples + 1):
                p = i / num_samples
                theta = p * final_theta + self.player_angle
                r = a * (p * final_theta)
                x = self.CENTER[0] + r * math.cos(theta)
                y = self.CENTER[1] + r * math.sin(theta)
                points.append((x, y))
        
        return points

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = (speed * math.cos(angle), speed * math.sin(angle))
            lifespan = self.np_random.integers(20, 50)
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': self.np_random.uniform(3, 8),
                'color': self.COLOR_PARTICLE
            })

    @staticmethod
    def _rotate_point(point, angle, center):
        x, y = point
        cx, cy = center
        new_x = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
        new_y = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        return new_x, new_y

    @staticmethod
    def _point_in_polygon(point, polygon_verts):
        x, y = point
        n = len(polygon_verts)
        inside = False
        p1x, p1y = polygon_verts[0]
        for i in range(n + 1):
            p2x, p2y = polygon_verts[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the autograder
    # but is useful for testing and debugging.
    
    # Un-comment the next line to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # The __init__ call for GameEnv now correctly handles pygame.init()
    # so we can create the display here.
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Spiral Gauntlet")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Action Mapping for Human Play ---
            movement = 0 # no-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            if keys[pygame.K_DOWN]: movement = 2
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0

            # --- Rendering ---
            # The observation is already a rendered frame
            # We just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Outcome: {env.game_outcome}")
                # Wait for a moment before auto-resetting or quitting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            clock.tick(GameEnv.FPS)
            
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because the environment is running in headless mode.")
        print("To run with a display, comment out `os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")` at the top of the file.")
    finally:
        env.close()