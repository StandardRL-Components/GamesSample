import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:25.977064
# Source Brief: brief_00724.md
# Brief Index: 724
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Guide a laser beam through a maze of mirrors to hit the target before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to steer the light beam towards the target."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = int(TIME_LIMIT_SECONDS * FPS * 1.5) # 90 seconds total buffer

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_MIRROR = (150, 150, 160)
    COLOR_TARGET = (255, 255, 255)
    COLOR_UI = (220, 220, 220)
    COLOR_BEAM_START = np.array([50, 150, 255])  # Blue
    COLOR_BEAM_END = np.array([255, 50, 50])    # Red

    # Game Mechanics
    INITIAL_BEAM_SPEED = 2.0
    MAX_BEAM_SPEED = 6.0
    TURN_RATE = 4.0  # degrees per step
    BEAM_TRAIL_LENGTH = 75
    SPEED_INCREASE_INTERVAL = 15 * 30 # Use 30fps for game logic ticks
    SPEED_INCREASE_FACTOR = 1.10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode

        # Initialize state variables (to be properly set in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.hit_target = False
        
        self.beam_pos = pygame.Vector2(0, 0)
        self.beam_vel = pygame.Vector2(0, 0)
        self.beam_speed = 0.0
        self.beam_trail = deque(maxlen=self.BEAM_TRAIL_LENGTH)
        
        self.target_pos = pygame.Vector2(0, 0)
        self.target_size = 0
        
        self.mirrors = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.hit_target = False

        # Beam setup
        self.beam_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        initial_angle = self.np_random.uniform(0, 360)
        self.beam_speed = self.INITIAL_BEAM_SPEED
        self.beam_vel = pygame.Vector2(self.beam_speed, 0).rotate(initial_angle)
        self.beam_trail = deque(maxlen=self.BEAM_TRAIL_LENGTH)
        self.beam_trail.append(self.beam_pos.copy())

        # Procedural generation
        self.target_pos, self.target_size = self._generate_target()
        self.mirrors = self._generate_mirrors()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        self._handle_actions(action)
        
        step_reward = self._update_beam()
        reward += step_reward
        
        self._update_particles()

        # Survival reward
        if self.steps % 10 == 0:
            reward += 0.01 # Reduced to not overpower other rewards

        # Difficulty scaling
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.beam_speed = min(self.MAX_BEAM_SPEED, self.beam_speed * self.SPEED_INCREASE_FACTOR)

        # Termination checks
        terminated = self.game_over
        time_ran_out = self._get_time_left() <= 0
        
        if time_ran_out and not terminated:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement = action[0]
        
        target_angle = None
        if movement == 1: target_angle = -90  # Up
        elif movement == 2: target_angle = 90   # Down
        elif movement == 3: target_angle = 180  # Left
        elif movement == 4: target_angle = 0    # Right

        if target_angle is not None:
            current_angle = self.beam_vel.angle_to(pygame.Vector2(1, 0))
            
            # Find shortest path to turn
            angle_diff = (target_angle - current_angle + 180) % 360 - 180
            turn = np.clip(angle_diff, -self.TURN_RATE, self.TURN_RATE)
            
            self.beam_vel.rotate_ip(turn)

    def _update_beam(self):
        reward = 0.0
        
        self.beam_vel.scale_to_length(self.beam_speed)
        start_pos = self.beam_pos.copy()
        end_pos = self.beam_pos + self.beam_vel

        # --- Collision Detection ---
        all_reflectors = self.mirrors + self._get_screen_boundaries()
        intersections = []

        for p1, p2 in all_reflectors:
            intersect_point = self._line_intersect(start_pos, end_pos, p1, p2)
            if intersect_point:
                dist_sq = start_pos.distance_squared_to(intersect_point)
                intersections.append((dist_sq, intersect_point, pygame.Vector2(p2) - pygame.Vector2(p1)))

        if intersections:
            # Find the closest intersection
            _, intersect_pos, mirror_vec = min(intersections, key=lambda x: x[0])
            
            # Move to collision point
            self.beam_pos = intersect_pos
            
            # Reflect velocity
            normal = mirror_vec.rotate(90).normalize()
            self.beam_vel.reflect_ip(normal)
            
            # Move for the rest of the frame
            remaining_dist = (end_pos - intersect_pos).length()
            self.beam_vel.scale_to_length(self.beam_speed) # Ensure speed is correct
            self.beam_pos += self.beam_vel.normalize() * remaining_dist

            self._spawn_particles(intersect_pos, 20)
            reward += 1.0
        else:
            self.beam_pos = end_pos

        self.beam_trail.append(self.beam_pos.copy())

        # --- Target Collision ---
        if self.beam_pos.distance_to(self.target_pos) < self.target_size:
            self.game_over = True
            self.hit_target = True
            reward += 100.0
            self._spawn_particles(self.target_pos, 50, (255, 255, 100))

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1    # lifespan--

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render mirrors
        for p1, p2 in self.mirrors:
            pygame.draw.aaline(self.screen, self.COLOR_MIRROR, p1, p2, 2)

        # Render target with glow
        for i in range(10, 0, -1):
            alpha = 100 - i * 10
            color = (self.COLOR_TARGET[0], self.COLOR_TARGET[1], self.COLOR_TARGET[2], alpha)
            s = pygame.Surface((self.target_size*2 + i*2, self.target_size*2 + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=int(i*0.8))
            self.screen.blit(s, (int(self.target_pos.x - self.target_size - i), int(self.target_pos.y - self.target_size - i)))
        pygame.draw.rect(self.screen, self.COLOR_TARGET, (self.target_pos.x - self.target_size, self.target_pos.y - self.target_size, self.target_size*2, self.target_size*2), border_radius=2)

        # Render beam trail
        if len(self.beam_trail) > 1:
            speed_ratio = np.clip((self.beam_speed - self.INITIAL_BEAM_SPEED) / (self.MAX_BEAM_SPEED - self.INITIAL_BEAM_SPEED), 0, 1)
            base_color = self.COLOR_BEAM_START * (1 - speed_ratio) + self.COLOR_BEAM_END * speed_ratio
            
            points = list(self.beam_trail)
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                life_ratio = i / self.BEAM_TRAIL_LENGTH
                color = base_color * life_ratio
                color = np.clip(color, 0, 255)
                
                if i > self.BEAM_TRAIL_LENGTH - 10:
                    color = np.clip(color * 1.5, 0, 255)
                
                try:
                    pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), max(1, int(life_ratio * 3)))
                except (TypeError, ValueError):
                    pass

        # Render particles
        for pos, vel, life, color in self.particles:
            size = int((life / 20.0) * 3)
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), size)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_left = self._get_time_left()
        time_color = (255, 80, 80) if time_left < 10 else self.COLOR_UI
        time_text = self.font.render(f"TIME: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self._get_time_left(),
            "beam_speed": self.beam_speed
        }

    # --- Helper Functions ---
    def _get_time_left(self):
        # Assuming game logic runs at 30fps for time calculation
        return max(0, self.TIME_LIMIT_SECONDS - (self.steps / 30.0))

    def _generate_target(self):
        padding = 50
        pos = pygame.Vector2(
            self.np_random.uniform(padding, self.WIDTH - padding),
            self.np_random.uniform(padding, self.HEIGHT - padding)
        )
        while pos.distance_to(pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)) < 150:
            pos = pygame.Vector2(
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            )
        size = 8
        return pos, size

    def _generate_mirrors(self):
        mirrors = []
        num_mirrors = self.np_random.integers(5, 9)
        padding = 20
        for _ in range(num_mirrors):
            length = self.np_random.uniform(50, 150)
            angle = self.np_random.uniform(0, 360)
            center = pygame.Vector2(
                self.np_random.uniform(padding + length/2, self.WIDTH - padding - length/2),
                self.np_random.uniform(padding + length/2, self.HEIGHT - padding - length/2)
            )
            
            p1 = center + pygame.Vector2(length / 2, 0).rotate(angle)
            p2 = center - pygame.Vector2(length / 2, 0).rotate(angle)
            
            if p1.distance_to(self.target_pos) > 50 and p2.distance_to(self.target_pos) > 50:
                 mirrors.append(( (p1.x, p1.y), (p2.x, p2.y) ))
        return mirrors

    def _get_screen_boundaries(self):
        return [
            ((0, 0), (self.WIDTH, 0)),
            ((self.WIDTH, 0), (self.WIDTH, self.HEIGHT)),
            ((self.WIDTH, self.HEIGHT), (0, self.HEIGHT)),
            ((0, self.HEIGHT), (0, 0))
        ]

    def _spawn_particles(self, pos, count, color=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            lifespan = self.np_random.integers(15, 30)
            p_color = color if color is not None else (
                self.np_random.integers(180, 256),
                self.np_random.integers(180, 256),
                self.np_random.integers(100, 200)
            )
            self.particles.append([pos.copy(), vel, lifespan, p_color])

    def _line_intersect(self, p1, p2, p3, p4):
        v1, v2, v3, v4 = pygame.Vector2(p1), pygame.Vector2(p2), pygame.Vector2(p3), pygame.Vector2(p4)
        
        r = v2 - v1
        s = v4 - v3
        
        r_cross_s = r.cross(s)
        if r_cross_s == 0:
            return None

        q_minus_p = v3 - v1
        t = q_minus_p.cross(s) / r_cross_s
        u = q_minus_p.cross(r) / r_cross_s

        if 0 < t < 1 and 0 < u < 1:
            intersection_point = v1 + t * r
            return intersection_point
        return None

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-enable display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Light Beam Maze")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0.0
    
    # --- Main Game Loop ---
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                print("--- Game Reset ---")

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()