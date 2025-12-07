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
        "Controls: Arrow keys to draw track from the cursor. "
        "Shift to move cursor to the rider. Space to extend the last line."
    )

    game_description = (
        "Draw a track for a physics-based rider to reach the finish line. "
        "Each line you draw becomes a solid surface. Plan your path carefully to guide the rider to victory."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.W, self.H = 640, 400
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (35, 45, 55)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_RIDER = (0, 150, 255)
        self.COLOR_RIDER_HEAD = (100, 200, 255)
        self.COLOR_START = (0, 255, 100)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_CHECKPOINT = (255, 200, 0)
        self.COLOR_SPARK = (255, 220, 180)
        self.COLOR_UI_TEXT = (230, 230, 230)

        # Physics constants
        self.GRAVITY = pygame.Vector2(0, 0.3)
        self.AIR_RESISTANCE = 0.998
        self.FRICTION = 0.9
        self.RIDER_RADIUS = 8
        self.LINE_SEGMENT_LENGTH = 15
        self.MAX_STEPS = 2000

        # All state variables are initialized in reset()
        self.rider_pos = None
        self.rider_vel = None
        self.track_segments = None
        self.drawing_cursor = None
        self.last_drawn_segment = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.checkpoint_reached = None
        self.start_pos = None
        self.finish_pos = None
        self.checkpoint_pos = None
        self.finish_rect = None
        self.checkpoint_rect = None

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.checkpoint_reached = False
        
        self.start_pos = pygame.Vector2(80, self.H / 2)
        self.finish_pos = pygame.Vector2(self.W - 80, self.H / 2)
        self.checkpoint_pos = self.start_pos.lerp(self.finish_pos, 0.5)

        self.finish_rect = pygame.Rect(self.finish_pos.x - 10, 0, 20, self.H)
        self.checkpoint_rect = pygame.Rect(self.checkpoint_pos.x - 5, 0, 10, self.H)

        self.rider_pos = self.start_pos.copy()
        self.rider_vel = pygame.Vector2(0, 0)
        
        self.track_segments = []
        # FIX: Add a starting platform to prevent the rider from falling immediately.
        platform_y = self.start_pos.y + self.RIDER_RADIUS + 2
        p1 = pygame.Vector2(self.start_pos.x - 30, platform_y)
        p2 = pygame.Vector2(self.start_pos.x + 30, platform_y)
        self.track_segments.append((p1, p2))

        self.drawing_cursor = self.rider_pos.copy()
        self.last_drawn_segment = None
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_drawing(movement, space_held, shift_held)
        
        prev_dist_to_finish = self.rider_pos.distance_to(self.finish_pos)

        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        
        reward = self._calculate_reward(prev_dist_to_finish)
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_drawing(self, movement, space_held, shift_held):
        if shift_held:
            # Move cursor to rider and break the current line chain
            self.drawing_cursor = self.rider_pos.copy()
            self.last_drawn_segment = None
            return

        drawn = False
        new_end_point = self.drawing_cursor.copy()

        if movement > 0:
            # Draw a new segment based on arrow key direction
            if movement == 1: new_end_point.y -= self.LINE_SEGMENT_LENGTH  # Up
            elif movement == 2: new_end_point.y += self.LINE_SEGMENT_LENGTH  # Down
            elif movement == 3: new_end_point.x -= self.LINE_SEGMENT_LENGTH  # Left
            elif movement == 4: new_end_point.x += self.LINE_SEGMENT_LENGTH  # Right
            drawn = True
        elif space_held and self.last_drawn_segment:
            # Extend the last drawn segment
            p1, p2 = self.last_drawn_segment
            direction = (p2 - p1).normalize() if (p2 - p1).length() > 0 else pygame.Vector2(1, 0)
            new_end_point = self.drawing_cursor + direction * self.LINE_SEGMENT_LENGTH
            drawn = True

        if drawn:
            # Clamp new point to screen bounds
            new_end_point.x = max(0, min(self.W, new_end_point.x))
            new_end_point.y = max(0, min(self.H, new_end_point.y))

            if self.drawing_cursor.distance_to(new_end_point) > 1: # Avoid zero-length lines
                new_segment = (self.drawing_cursor.copy(), new_end_point.copy())
                self.track_segments.append(new_segment)
                self.last_drawn_segment = new_segment
                self.drawing_cursor = new_end_point.copy()
                # Limit total track segments to avoid performance degradation
                if len(self.track_segments) > 200:
                    self.track_segments.pop(0)

    def _update_physics(self):
        if self.game_over: return

        # Apply gravity and air resistance
        self.rider_vel += self.GRAVITY
        self.rider_vel *= self.AIR_RESISTANCE
        self.rider_pos += self.rider_vel

        # Collision with track segments
        for p1, p2 in self.track_segments:
            d_sq = (p2 - p1).length_squared()
            if d_sq == 0: continue
            
            t = ((self.rider_pos - p1).dot(p2 - p1)) / d_sq
            t = max(0, min(1, t))
            closest_point = p1.lerp(p2, t)

            dist_vec = self.rider_pos - closest_point
            if dist_vec.length_squared() < self.RIDER_RADIUS ** 2:
                # Collision detected
                dist = dist_vec.length()
                penetration = self.RIDER_RADIUS - dist
                
                # Push rider out of the line
                if dist > 0:
                    self.rider_pos += dist_vec.normalize() * penetration
                else: # Rider is exactly on the line
                    self.rider_pos += (p2-p1).rotate(90).normalize() * penetration

                # Calculate collision response (reflection and friction)
                normal = dist_vec.normalize() if dist > 0 else (p2-p1).rotate(90).normalize()
                self.rider_vel -= 2 * self.rider_vel.dot(normal) * normal
                self.rider_vel *= self.FRICTION

                # Create spark particles
                if self.rider_vel.length() > 1.5:
                    self._create_particles(self.rider_pos, 3, self.COLOR_SPARK, 2)
                    # sfx: metal scrape sound

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['lifespan'] / p['max_lifespan'] * 3)

    def _calculate_reward(self, prev_dist_to_finish):
        if self.game_over:
            if self.win: return 100.0
            elif not (0 < self.rider_pos.x < self.W and 0 < self.rider_pos.y < self.H): return -50.0

        reward = 0
        
        # Reward for getting closer to the finish
        current_dist_to_finish = self.rider_pos.distance_to(self.finish_pos)
        reward += (prev_dist_to_finish - current_dist_to_finish) * 0.1

        # Checkpoint reward
        rider_rect = pygame.Rect(self.rider_pos.x - self.RIDER_RADIUS, self.rider_pos.y - self.RIDER_RADIUS, self.RIDER_RADIUS*2, self.RIDER_RADIUS*2)
        if not self.checkpoint_reached and self.checkpoint_rect.colliderect(rider_rect):
            self.checkpoint_reached = True
            reward += 5.0
        
        return reward

    def _check_termination(self):
        if self.game_over: return True

        rider_rect = pygame.Rect(self.rider_pos.x - self.RIDER_RADIUS, self.rider_pos.y - self.RIDER_RADIUS, self.RIDER_RADIUS*2, self.RIDER_RADIUS*2)

        # Win condition
        if self.finish_rect.colliderect(rider_rect):
            self.game_over = True
            self.win = True
            return True

        # Loss conditions
        if not (0 < self.rider_pos.x < self.W and 0 < self.rider_pos.y < self.H):
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
        # Draw background grid
        for x in range(0, self.W, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.W, y))

        # Draw start, finish, and checkpoint zones
        pygame.draw.rect(self.screen, self.COLOR_START, (self.start_pos.x - 10, 0, 20, self.H), 0, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_rect, 0, border_radius=5)
        if not self.checkpoint_reached:
            pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT, self.checkpoint_rect, 0, border_radius=3)
        
        # Draw track segments
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)

        # Draw drawing cursor
        if not self.game_over:
            pygame.gfxdraw.filled_circle(self.screen, int(self.drawing_cursor.x), int(self.drawing_cursor.y), 4, (*self.COLOR_TRACK, 100))
            pygame.gfxdraw.aacircle(self.screen, int(self.drawing_cursor.x), int(self.drawing_cursor.y), 4, (*self.COLOR_TRACK, 150))
        
        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Draw rider
        rider_x, rider_y = int(self.rider_pos.x), int(self.rider_pos.y)
        head_offset = self.rider_vel.normalize() * (self.RIDER_RADIUS * 0.5) if self.rider_vel.length() > 0 else pygame.Vector2(0, -1) * (self.RIDER_RADIUS * 0.5)
        head_pos = self.rider_pos + head_offset
        
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, int(head_pos.x), int(head_pos.y), 4, self.COLOR_RIDER_HEAD)
        pygame.gfxdraw.aacircle(self.screen, int(head_pos.x), int(head_pos.y), 4, self.COLOR_RIDER_HEAD)
        
        # Draw speed lines
        speed = self.rider_vel.length()
        if speed > 5:
            num_lines = min(5, int(speed / 3))
            for _ in range(num_lines):
                offset = pygame.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
                start = self.rider_pos - self.rider_vel.normalize() * random.uniform(10, 20) + offset
                end = start - self.rider_vel.normalize() * random.uniform(5, 10)
                pygame.draw.aaline(self.screen, self.COLOR_RIDER_HEAD, start, end)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.W - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                message = "TIME UP"
                color = self.COLOR_CHECKPOINT
            else:
                message = "FINISH!" if self.win else "CRASHED"
                color = self.COLOR_START if self.win else self.COLOR_FINISH
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
            "checkpoint_reached": self.checkpoint_reached,
        }

    def _create_particles(self, pos, count, color, speed_range):
        for _ in range(count):
            lifespan = random.randint(10, 20)
            self.particles.append({
                'pos': pos.copy() + pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                'vel': pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(0.5, speed_range),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'radius': 3
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Un-set the headless environment variable for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Track Rider")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling (for quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(60)

    print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()