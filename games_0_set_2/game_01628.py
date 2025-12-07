
# Generated: 2025-08-27T17:44:55.951140
# Source Brief: brief_01628.md
# Brief Index: 1628

        
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
        "Controls: Use arrow keys to accelerate. Hold Space for a speed boost. Hold Shift to brake."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade racer. Navigate the track, avoid obstacles, and complete three laps against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    LAP_TIME_LIMIT_SECONDS = 60
    TOTAL_LAPS = 3

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_TRACK = (180, 180, 180)
    COLOR_TRACK_BORDER = (220, 220, 220)
    COLOR_OBSTACLE = (0, 150, 255)
    COLOR_OBSTACLE_OUTLINE = (100, 200, 255)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_OUTLINE = (255, 150, 150)
    COLOR_CHECKPOINT = (50, 200, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SPARK = (255, 220, 100)
    COLOR_BOOST = (255, 150, 0)
    COLOR_BRAKE = (200, 200, 200)

    # Physics
    ACCELERATION = 0.25
    BOOST_ACCELERATION = 0.6
    MAX_SPEED = 8.0
    DRAG = 0.96
    BRAKE_DRAG = 0.85

    # Game Elements
    PLAYER_SIZE = (12, 25)
    OBSTACLE_RADIUS = 10
    TRACK_WIDTH = 60
    INITIAL_OBSTACLES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20, bold=True)

        self.player_pos = None
        self.player_vel = None
        self.lap = None
        self.lap_time_steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.obstacles = None
        self.particles = None
        self.track_nodes = None
        self.next_checkpoint_index = None
        self.last_step_reward = None
        self.near_miss_cooldown = None
        
        self.reset()
        self.validate_implementation()


    def _generate_track(self):
        w, h = self.WIDTH * 2, self.HEIGHT * 2
        corner_radius = 150
        self.track_nodes = [
            pygame.Vector2(corner_radius, 0),
            pygame.Vector2(w - corner_radius, 0),
            pygame.Vector2(w, corner_radius),
            pygame.Vector2(w, h - corner_radius),
            pygame.Vector2(w - corner_radius, h),
            pygame.Vector2(corner_radius, h),
            pygame.Vector2(0, h - corner_radius),
            pygame.Vector2(0, corner_radius),
        ]
        self.start_pos = pygame.Vector2(corner_radius, -self.TRACK_WIDTH / 2)
        self.start_angle = 0 # Radians


    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = self.INITIAL_OBSTACLES + self.lap - 1
        
        for _ in range(num_obstacles):
            while True:
                segment_idx = self.np_random.integers(0, len(self.track_nodes))
                p1 = self.track_nodes[segment_idx]
                p2 = self.track_nodes[(segment_idx + 1) % len(self.track_nodes)]
                
                t = self.np_random.random()
                point_on_line = p1.lerp(p2, t)
                
                normal = (p2 - p1).normalize().rotate(90)
                offset = self.np_random.uniform(-self.TRACK_WIDTH * 0.8, self.TRACK_WIDTH * 0.8)
                
                pos = point_on_line + normal * offset
                
                # Ensure it's not too close to the start line
                if pos.distance_to(self.start_pos) < 200:
                    continue
                
                # Ensure it's not too close to other obstacles
                too_close = any(pos.distance_to(obs_pos) < self.OBSTACLE_RADIUS * 4 for obs_pos in self.obstacles)
                if not too_close:
                    self.obstacles.append(pos)
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_track()
        
        self.player_pos = self.start_pos.copy()
        self.player_vel = pygame.Vector2(0, 0)
        
        self.lap = 1
        self.lap_time_steps = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.particles = []
        self._generate_obstacles()
        
        self.next_checkpoint_index = 1
        self.last_step_reward = 0
        self.near_miss_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_physics(movement, space_held, shift_held)
        reward = self._update_game_state()
        
        self.score += reward
        self.last_step_reward = reward
        self.steps += 1
        
        terminated = self.game_over or self.win
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self, movement, space_held, shift_held):
        accel = pygame.Vector2(0, 0)
        
        if movement == 1: accel.y = -self.ACCELERATION
        elif movement == 2: accel.y = self.ACCELERATION
        elif movement == 3: accel.x = -self.ACCELERATION
        elif movement == 4: accel.x = self.ACCELERATION

        if space_held:
            # Sfx: Boost sound
            boost_dir = self.player_vel.normalize() if self.player_vel.length() > 0.1 else pygame.Vector2(0, -1).rotate(-math.degrees(self._get_player_angle()))
            accel += boost_dir * self.BOOST_ACCELERATION
            if self.np_random.random() < 0.5:
                self._add_particles(1, self.COLOR_BOOST, life=10, speed=3, size=(2,4))

        self.player_vel += accel
        
        drag = self.BRAKE_DRAG if shift_held else self.DRAG
        if shift_held and self.player_vel.length() > 1.0:
            # Sfx: Tire screech
            self._add_particles(1, self.COLOR_BRAKE, life=15, speed=1, size=(3,3))

        self.player_vel *= drag
        
        if self.player_vel.length() > self.MAX_SPEED:
            self.player_vel.scale_to_length(self.MAX_SPEED)
            
        self.player_pos += self.player_vel

    def _update_game_state(self):
        reward = 0
        self.lap_time_steps += 1
        if self.near_miss_cooldown > 0:
            self.near_miss_cooldown -= 1

        # Reward for forward progress
        track_dir = self._get_track_direction(self.player_pos)
        progress = self.player_vel.dot(track_dir)
        reward += progress * 0.01 # Scaled down reward for movement

        # Check for collision
        for obs_pos in self.obstacles:
            dist = self.player_pos.distance_to(obs_pos)
            if dist < self.OBSTACLE_RADIUS + max(self.PLAYER_SIZE) / 2:
                # Sfx: Crash sound
                self.game_over = True
                reward -= 100
                self._add_particles(50, self.COLOR_SPARK, life=40, speed=5, size=(1, 3))
                break
            elif self.near_miss_cooldown == 0 and dist < self.OBSTACLE_RADIUS + self.TRACK_WIDTH / 2:
                # Sfx: Spark sound
                reward -= 0.5
                self.near_miss_cooldown = 15 # 0.5 sec cooldown
                self._add_particles(10, self.COLOR_SPARK, life=20, speed=3, size=(1,2))

        # Check for checkpoint crossing
        p1 = self.player_pos - self.player_vel
        p2 = self.player_pos
        checkpoint_line_p1 = self.track_nodes[self.next_checkpoint_index]
        checkpoint_line_p2 = checkpoint_line_p1 + self._get_track_direction(checkpoint_line_p1).rotate(90) * self.TRACK_WIDTH * 2

        if self._line_segment_intersection(p1, p2, checkpoint_line_p1, checkpoint_line_p2):
            # Sfx: Checkpoint sound
            self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(self.track_nodes)
            reward += 5
            
            # Lap complete
            if self.next_checkpoint_index == 1: # Crossed finish line (which is checkpoint 0, next is 1)
                self.lap += 1
                if self.lap > self.TOTAL_LAPS:
                    self.win = True
                    reward += 100
                else:
                    # Sfx: Lap complete fanfare
                    reward += 100
                    self.lap_time_steps = 0
                    self._generate_obstacles()

        # Check for lap timeout
        if self.lap_time_steps > self.LAP_TIME_LIMIT_SECONDS * self.FPS:
            self.game_over = True
            reward -= 100
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2) - self.player_pos
        
        self._render_track(cam_offset)
        self._render_obstacles(cam_offset)
        self._render_particles(cam_offset)
        self._render_player(cam_offset)
        
        if self.game_over or self.win:
            self._render_end_screen()

    def _render_track(self, offset):
        # Draw track fill
        screen_nodes = [node + offset for node in self.track_nodes]
        pygame.draw.polygon(self.screen, self.COLOR_TRACK, screen_nodes)

        # Draw borders and checkpoints
        for i in range(len(self.track_nodes)):
            p1 = self.track_nodes[i]
            p2 = self.track_nodes[(i + 1) % len(self.track_nodes)]
            sp1, sp2 = p1 + offset, p2 + offset
            
            # Finish line
            if i == len(self.track_nodes) - 1:
                self._draw_checkered_line(sp2, sp1, 10, self.TRACK_WIDTH * 2)
            # Checkpoints
            elif i + 1 == self.next_checkpoint_index:
                 pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, sp1, sp2, 5)

        pygame.draw.aalines(self.screen, self.COLOR_TRACK_BORDER, True, screen_nodes, 2)


    def _render_obstacles(self, offset):
        for obs_pos in self.obstacles:
            screen_pos = obs_pos + offset
            if -20 < screen_pos.x < self.WIDTH + 20 and -20 < screen_pos.y < self.HEIGHT + 20:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE_OUTLINE)

    def _render_player(self, offset):
        player_screen_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        angle = self._get_player_angle()
        
        player_surface = pygame.Surface(self.PLAYER_SIZE, pygame.SRCALPHA)
        player_surface.fill(self.COLOR_PLAYER)
        pygame.draw.rect(player_surface, self.COLOR_PLAYER_OUTLINE, player_surface.get_rect(), 1)
        
        rotated_surface = pygame.transform.rotate(player_surface, math.degrees(angle))
        new_rect = rotated_surface.get_rect(center=player_screen_pos)
        
        self.screen.blit(rotated_surface, new_rect.topleft)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p['pos'] + offset
            size = max(1, int(p['size'][0] * (p['life'] / p['max_life'])))
            pygame.draw.circle(self.screen, p['color'], (int(pos.x), int(pos.y)), size)
    
    def _render_ui(self):
        lap_text = self.font_small.render(f"LAP: {min(self.lap, self.TOTAL_LAPS)}/{self.TOTAL_LAPS}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (self.WIDTH - lap_text.get_width() - 10, 10))
        
        time_left = max(0, self.LAP_TIME_LIMIT_SECONDS - self.lap_time_steps / self.FPS)
        time_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(time_text, (10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        end_text_str = "RACE COMPLETE" if self.win else "GAME OVER"
        end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
        text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        
        overlay.blit(end_text, text_rect)
        self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "lap_time": self.lap_time_steps / self.FPS,
        }

    def _get_player_angle(self):
        if self.player_vel.length() > 0.1:
            return -self.player_vel.angle_to(pygame.Vector2(0, -1)) * (math.pi / 180.0)
        return self.start_angle

    def _add_particles(self, count, color, life, speed, size):
        angle = self._get_player_angle()
        for _ in range(count):
            p_angle = angle + math.pi + self.np_random.uniform(-0.5, 0.5)
            p_vel = pygame.Vector2(math.sin(p_angle), math.cos(p_angle)) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': p_vel,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'size': size
            })

    def _get_track_direction(self, pos):
        min_dist = float('inf')
        best_proj = None
        
        for i in range(len(self.track_nodes)):
            p1 = self.track_nodes[i]
            p2 = self.track_nodes[(i + 1) % len(self.track_nodes)]
            
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = pos - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t))
            
            proj_point = p1 + t * line_vec
            dist = pos.distance_to(proj_point)
            
            if dist < min_dist:
                min_dist = dist
                best_proj = line_vec.normalize()
                
        return best_proj if best_proj else pygame.Vector2(1,0)

    def _line_segment_intersection(self, p1, p2, p3, p4):
        # Check if two line segments intersect
        v1 = p2 - p1
        v2 = p4 - p3
        cross = v1.cross(v2)
        if abs(cross) < 1e-6: return False # Parallel lines
        
        t = (p3 - p1).cross(v2) / cross
        u = (p3 - p1).cross(v1) / cross
        
        return 0 <= t <= 1 and 0 <= u <= 1

    def _draw_checkered_line(self, p1, p2, num_checks, thickness):
        line_vec = p2 - p1
        if line_vec.length() == 0: return
        
        normal = line_vec.normalize().rotate(90)
        
        for i in range(num_checks):
            color = (255, 255, 255) if i % 2 == 0 else (50, 50, 50)
            start = p1.lerp(p2, i / num_checks)
            end = p1.lerp(p2, (i + 1) / num_checks)
            
            points = [
                start - normal * thickness / 2,
                end - normal * thickness / 2,
                end + normal * thickness / 2,
                start + normal * thickness / 2,
            ]
            pygame.draw.polygon(self.screen, color, points)

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run this environment, you can use the following code:
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}. Press 'R' to restart.")
            
        clock.tick(GameEnv.FPS)

    pygame.quit()