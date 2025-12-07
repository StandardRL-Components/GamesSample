import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:01:02.912753
# Source Brief: brief_02993.md
# Brief Index: 2993
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Direct a stream of particles into a target goal. Adjust the angle and emission rate to guide all particles home before time runs out."
    user_guide = "Controls: Use ↑/↓ arrow keys to change the particle emission rate and ←/→ to steer the stream."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (100, 100, 120)
    COLOR_GOAL = (50, 255, 150)
    COLOR_GOAL_GLOW = (50, 255, 150, 50)
    COLOR_EMITTER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_WARN = (255, 100, 100)

    # Particle Colors (rate 1 to 5)
    PARTICLE_COLORS = [
        (100, 150, 255),  # Blue (Slow)
        (80, 220, 255),   # Cyan
        (100, 255, 100),  # Green
        (255, 255, 100),  # Yellow
        (255, 100, 100),  # Red (Fast)
    ]

    # Gameplay Parameters
    TOTAL_PARTICLES = 50
    EMISSION_RATE_LEVELS = {
        1: 30,  # frames per particle
        2: 20,
        3: 12,
        4: 8,
        5: 5,
    }
    PARTICLE_SPEED = 2.5
    PARTICLE_RADIUS = 3
    PARTICLE_TRAIL_LENGTH = 10
    EMITTER_TURN_RATE = math.pi / 48
    GOAL_RADIUS = 35

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_goal = pygame.font.SysFont("monospace", 16, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False
        self.time_remaining = 0
        self.emitter_pos = None
        self.emitter_angle = None
        self.emission_rate_level = 0
        self.emission_cooldown = 0
        self.particles = []
        self.particles_spawned = 0
        self.particles_in_goal = 0
        self.goal_pos = None
        self.walls = []
        
        # Initialize state variables
        self.reset()

        # --- Critical Self-Check ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        # Emitter
        self.emitter_pos = np.array([self.WIDTH / 2, 60.0])
        self.emitter_angle = math.pi / 2  # Pointing straight down
        self.emission_rate_level = 1
        self.emission_cooldown = 0

        # Particles
        self.particles = []
        self.particles_spawned = 0
        self.particles_in_goal = 0

        # Goal
        self.goal_pos = np.array([self.WIDTH / 2, self.HEIGHT - 70.0])

        # Level layout (walls)
        self.walls = [
            # Outer bounds
            ((0, 0), (self.WIDTH, 0)),
            ((0, 0), (0, self.HEIGHT)),
            ((self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT)),
            ((0, self.HEIGHT - 1), (self.WIDTH, self.HEIGHT - 1)),
            # Inner obstacle
            ((100, 200), (self.WIDTH - 100, 200)),
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        # --- Handle Player Input ---
        if movement == 1:  # Up: Increase emission rate
            self.emission_rate_level = min(5, self.emission_rate_level + 1)
        elif movement == 2:  # Down: Decrease emission rate
            self.emission_rate_level = max(1, self.emission_rate_level - 1)
        elif movement == 3:  # Left: Steer left
            self.emitter_angle -= self.EMITTER_TURN_RATE
        elif movement == 4:  # Right: Steer right
            self.emitter_angle += self.EMITTER_TURN_RATE
        
        # Clamp angle to prevent pointing upwards
        self.emitter_angle = np.clip(self.emitter_angle, math.pi * 0.1, math.pi * 0.9)

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # Spawn new particles
        self._spawn_particles()

        # Update existing particles
        reward += self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        if self.particles_in_goal == self.TOTAL_PARTICLES:
            # Victory
            terminated = True
            reward += 50 + (self.time_remaining / self.MAX_STEPS) * 50 # Bonus for time left
            # sfx: game_win_sound
        elif self.time_remaining <= 0:
            # Failure (Timeout)
            terminated = True
            reward -= 100
            # sfx: game_over_sound

        self.game_over = terminated
        self.cumulative_reward += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _spawn_particles(self):
        self.emission_cooldown -= 1
        if self.emission_cooldown <= 0 and self.particles_spawned < self.TOTAL_PARTICLES:
            self.emission_cooldown = self.EMISSION_RATE_LEVELS[self.emission_rate_level]
            self.particles_spawned += 1

            vel = np.array([math.cos(self.emitter_angle), math.sin(self.emitter_angle)]) * self.PARTICLE_SPEED
            
            # Add slight randomness to velocity
            vel += self.np_random.normal(0, 0.1, 2)
            
            new_particle = {
                "pos": self.emitter_pos.copy(),
                "vel": vel,
                "color": self.PARTICLE_COLORS[self.emission_rate_level - 1],
                "trail": deque(maxlen=self.PARTICLE_TRAIL_LENGTH),
            }
            self.particles.append(new_particle)
            # sfx: particle_spawn_sound

    def _update_particles(self):
        reward = 0
        particles_to_remove = []

        for i, p in enumerate(self.particles):
            old_dist_to_goal = np.linalg.norm(p["pos"] - self.goal_pos)
            
            # Update position and trail
            p["trail"].append(p["pos"].copy())
            p["pos"] += p["vel"]

            # Reward for getting closer to the goal
            new_dist_to_goal = np.linalg.norm(p["pos"] - self.goal_pos)
            if new_dist_to_goal < old_dist_to_goal:
                reward += 0.01 # Small continuous reward
            
            # Check for goal collision
            if new_dist_to_goal < self.GOAL_RADIUS:
                particles_to_remove.append(i)
                self.particles_in_goal += 1
                reward += 1.0
                # sfx: particle_goal_sound
                continue

            # Check for wall collision
            # Vertical walls
            if p["pos"][0] < self.PARTICLE_RADIUS or p["pos"][0] > self.WIDTH - self.PARTICLE_RADIUS:
                p["vel"][0] *= -1
                p["pos"][0] = np.clip(p["pos"][0], self.PARTICLE_RADIUS, self.WIDTH - self.PARTICLE_RADIUS)
                # sfx: particle_bounce_sound
            # Horizontal walls
            if p["pos"][1] < self.PARTICLE_RADIUS or p["pos"][1] > self.HEIGHT - self.PARTICLE_RADIUS:
                p["vel"][1] *= -1
                p["pos"][1] = np.clip(p["pos"][1], self.PARTICLE_RADIUS, self.HEIGHT - self.PARTICLE_RADIUS)
                # sfx: particle_bounce_sound
            
            # Inner wall collision (simple AABB check)
            wall = self.walls[4] # The horizontal barrier
            wall_y = wall[0][1]
            wall_x1, wall_x2 = wall[0][0], wall[1][0]
            if wall_x1 < p["pos"][0] < wall_x2 and abs(p["pos"][1] - wall_y) < self.PARTICLE_RADIUS * 2:
                 p["vel"][1] *= -1
                 # sfx: particle_bounce_sound

        # Remove particles that reached the goal
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.line(self.screen, self.COLOR_WALL, wall[0], wall[1], 2)

        # Render goal with glow effect
        goal_pos_int = (int(self.goal_pos[0]), int(self.goal_pos[1]))
        for i in range(5):
            radius = self.GOAL_RADIUS + i * 4
            alpha = 1 - (i / 5)
            glow_color = (*self.COLOR_GOAL, int(50 * alpha))
            pygame.gfxdraw.filled_circle(self.screen, goal_pos_int[0], goal_pos_int[1], radius, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, goal_pos_int[0], goal_pos_int[1], self.GOAL_RADIUS, self.COLOR_GOAL)
        pygame.gfxdraw.aacircle(self.screen, goal_pos_int[0], goal_pos_int[1], self.GOAL_RADIUS, self.COLOR_GOAL)

        # Render particles and trails
        for p in self.particles:
            # Trail
            if len(p["trail"]) > 1:
                for i in range(len(p["trail"]) - 1):
                    alpha = (i / self.PARTICLE_TRAIL_LENGTH) * 0.8
                    color = (*p["color"], int(255 * alpha))
                    start_pos = (int(p["trail"][i][0]), int(p["trail"][i][1]))
                    end_pos = (int(p["trail"][i+1][0]), int(p["trail"][i+1][1]))
                    pygame.draw.line(self.screen, color, start_pos, end_pos, self.PARTICLE_RADIUS)

            # Particle
            pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, p["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, p["color"])

        # Render emitter
        emitter_pos_int = (int(self.emitter_pos[0]), int(self.emitter_pos[1]))
        p1 = (
            emitter_pos_int[0] + 10 * math.cos(self.emitter_angle),
            emitter_pos_int[1] + 10 * math.sin(self.emitter_angle),
        )
        p2 = (
            emitter_pos_int[0] + 8 * math.cos(self.emitter_angle + math.pi / 2),
            emitter_pos_int[1] + 8 * math.sin(self.emitter_angle + math.pi / 2),
        )
        p3 = (
            emitter_pos_int[0] + 8 * math.cos(self.emitter_angle - math.pi / 2),
            emitter_pos_int[1] + 8 * math.sin(self.emitter_angle - math.pi / 2),
        )
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_EMITTER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_EMITTER)

    def _render_ui(self):
        # Timer
        time_sec = self.time_remaining / self.FPS
        timer_text = f"TIME: {time_sec:.2f}"
        time_color = self.COLOR_TEXT if time_sec > 5 else self.COLOR_TEXT_WARN
        text_surf = self.font_ui.render(timer_text, True, time_color)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Particles in goal
        goal_text = f"{self.particles_in_goal}/{self.TOTAL_PARTICLES}"
        goal_surf = self.font_goal.render(goal_text, True, self.COLOR_TEXT)
        goal_rect = goal_surf.get_rect(center=(int(self.goal_pos[0]), int(self.goal_pos[1])))
        self.screen.blit(goal_surf, goal_rect)

        # Emission rate indicator
        rate_text = f"RATE:"
        rate_surf = self.font_ui.render(rate_text, True, self.COLOR_TEXT)
        self.screen.blit(rate_surf, (10, 10))
        for i in range(5):
            color = self.PARTICLE_COLORS[i] if i < self.emission_rate_level else self.COLOR_WALL
            bar_rect = pygame.Rect(rate_surf.get_width() + 15 + i * 15, 15, 10, 15)
            pygame.draw.rect(self.screen, color, bar_rect)

    def _get_info(self):
        return {
            "score": self.cumulative_reward,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "particles_in_goal": self.particles_in_goal,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Particle Stream")

    terminated = False
    total_reward = 0
    
    # Key mapping for manual control
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print("\n--- Manual Control ---")
    print("UP/DOWN: Change emission rate")
    print("LEFT/RIGHT: Steer particle stream")
    print("Q: Quit")

    while not terminated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display window
        pygame.surfarray.blit_array(env.screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"--- Game Over ---")
            print(f"Final Score: {info['score']:.2f}")
            print(f"Steps: {info['steps']}")
            print(f"Particles in Goal: {info['particles_in_goal']}/{env.TOTAL_PARTICLES}")
            
            # Wait for a moment before closing
            pygame.time.wait(3000)

    env.close()