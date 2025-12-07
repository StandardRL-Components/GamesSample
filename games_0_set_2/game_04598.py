
# Generated: 2025-08-28T02:53:09.721122
# Source Brief: brief_04598.md
# Brief Index: 4598

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Survive the drone swarm for as long as possible."
    )

    game_description = (
        "A top-down arcade survival game. Dodge procedurally generated drones across three increasingly difficult stages."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_HITS = 5
        self.STAGE_DURATION_SECONDS = 60
        self.TOTAL_STAGES = 3
        self.DIFFICULTY_INTERVAL_SECONDS = 10

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_BORDER = (128, 255, 200)
        self.COLOR_DRONE = (255, 50, 50)
        self.COLOR_DRONE_TRAIL = (100, 20, 20)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 150, 50)
        
        # Player settings
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5.0

        # Drone settings
        self.DRONE_SIZE = 12
        self.INITIAL_DRONE_COUNT = 3
        self.INITIAL_DRONE_SPEED = 1.5
        self.DRONE_COUNT_INCREASE = 1
        self.DRONE_SPEED_INCREASE = 0.2

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.drones = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_condition = None
        self.current_stage = None
        self.hits_in_stage = None
        self.steps_in_stage = None
        self.time_since_difficulty_increase = None
        self.current_drone_count = None
        self.current_drone_speed = None

        self.reset()
        
        # This must be the last call in __init__
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.drones = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.current_stage = 1
        self._reset_stage_state()

        for _ in range(self.current_drone_count):
            self._spawn_drone()

        return self._get_observation(), self._get_info()

    def _reset_stage_state(self):
        """Resets the state for the beginning of a new stage."""
        self.hits_in_stage = 0
        self.steps_in_stage = 0
        self.time_since_difficulty_increase = 0
        self.current_drone_count = self.INITIAL_DRONE_COUNT
        self.current_drone_speed = self.INITIAL_DRONE_SPEED
        self.drones.clear()
        self.particles.clear()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0.0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info(),
            )

        movement, _, _ = action
        reward = 0.0

        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        self._update_drones()
        self._update_particles()

        # --- Collision Detection ---
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        
        drones_to_remove = []
        for i, drone in enumerate(self.drones):
            if player_rect.colliderect(drone.get_rect()):
                drones_to_remove.append(i)
                self.hits_in_stage += 1
                reward -= 5.0
                # Sound: Player hit
                self._create_collision_effect(drone.pos)

        # Remove collided drones in reverse to avoid index errors
        for i in sorted(drones_to_remove, reverse=True):
            del self.drones[i]
            self._spawn_drone() # Respawn a new drone immediately

        # --- Update Timers and State ---
        self.steps += 1
        self.steps_in_stage += 1
        self.time_since_difficulty_increase += 1
        reward += 0.1  # Survival reward

        # --- Difficulty Scaling ---
        if self.time_since_difficulty_increase >= self.DIFFICULTY_INTERVAL_SECONDS * self.FPS:
            self.time_since_difficulty_increase = 0
            self.current_drone_count += self.DRONE_COUNT_INCREASE
            self.current_drone_speed += self.DRONE_SPEED_INCREASE
            # Ensure drone count matches target
            while len(self.drones) < self.current_drone_count:
                self._spawn_drone()

        # --- Check Termination Conditions ---
        terminated = False
        # Loss Condition
        if self.hits_in_stage >= self.MAX_HITS:
            terminated = True
            self.game_over = True
            # Sound: Game Over

        # Stage Clear / Win Condition
        if self.steps_in_stage >= self.STAGE_DURATION_SECONDS * self.FPS:
            reward += 10.0 # Stage complete bonus
            if self.current_stage >= self.TOTAL_STAGES:
                reward += 100.0 # Final win bonus
                terminated = True
                self.game_over = True
                self.win_condition = True
                # Sound: Victory
            else:
                # Advance to next stage
                self.current_stage += 1
                self._reset_stage_state()
                for _ in range(self.current_drone_count):
                    self._spawn_drone()
                # Sound: Stage Clear

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        half_size = self.PLAYER_SIZE / 2
        self.player_pos[0] = np.clip(self.player_pos[0], half_size, self.WIDTH - half_size)
        self.player_pos[1] = np.clip(self.player_pos[1], half_size, self.HEIGHT - half_size)

    def _spawn_drone(self):
        self.drones.append(Drone(self.WIDTH, self.HEIGHT, self.current_drone_speed, self.DRONE_SIZE, self.np_random))

    def _update_drones(self):
        drones_to_remove = []
        for i, drone in enumerate(self.drones):
            drone.update()
            if not drone.is_on_screen():
                drones_to_remove.append(i)
        
        for i in sorted(drones_to_remove, reverse=True):
            del self.drones[i]
            self._spawn_drone()

    def _create_collision_effect(self, position):
        # Sound: Explosion
        for _ in range(20):
            self.particles.append(Particle(position, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render drone trails
        for drone in self.drones:
            drone.draw_trail(self.screen, self.COLOR_DRONE_TRAIL)
        
        # Render particles
        for particle in self.particles:
            particle.draw(self.screen, self.COLOR_PARTICLE)

        # Render drones
        for drone in self.drones:
            drone.draw(self.screen, self.COLOR_DRONE)

        # Render player
        player_rect = pygame.Rect(
            int(self.player_pos[0] - self.PLAYER_SIZE / 2),
            int(self.player_pos[1] - self.PLAYER_SIZE / 2),
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_BORDER, player_rect, width=2, border_radius=2)

    def _render_ui(self):
        # Time remaining
        time_left = max(0, self.STAGE_DURATION_SECONDS - (self.steps_in_stage / self.FPS))
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage}/{self.TOTAL_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))

        # Hits
        hits_text = self.font_ui.render(f"HITS: {self.hits_in_stage}/{self.MAX_HITS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_text, (self.WIDTH / 2 - hits_text.get_width() / 2, self.HEIGHT - hits_text.get_height() - 10))

        # Game Over / Win Message
        if self.game_over:
            msg_text_str = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = self.COLOR_PLAYER if self.win_condition else self.COLOR_DRONE
            msg_surface = self.font_msg.render(msg_text_str, True, color)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = msg_rect.inflate(20, 20)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(bg_surface, bg_rect)
            
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "hits": self.hits_in_stage,
        }

    def close(self):
        pygame.quit()

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


class Drone:
    def __init__(self, screen_width, screen_height, speed, size, np_random):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed
        self.size = size
        self.np_random = np_random
        self.trail = collections.deque(maxlen=20)
        self._init_path()

    def _init_path(self):
        side = self.np_random.integers(0, 4)
        margin = self.size * 2
        
        self.path_type = self.np_random.choice(['sin', 'cos'])
        self.amplitude = self.np_random.uniform(20, self.screen_height / 4)
        self.frequency = self.np_random.uniform(0.01, 0.03)
        self.phase = self.np_random.uniform(0, 2 * math.pi)

        if side == 0:  # Left
            self.pos = np.array([-margin, self.np_random.uniform(0, self.screen_height)], dtype=float)
            self.base_line = self.pos[1]
            self.axis = 0 # moves along x
        elif side == 1:  # Right
            self.pos = np.array([self.screen_width + margin, self.np_random.uniform(0, self.screen_height)], dtype=float)
            self.base_line = self.pos[1]
            self.axis = 0
        elif side == 2:  # Top
            self.pos = np.array([self.np_random.uniform(0, self.screen_width), -margin], dtype=float)
            self.base_line = self.pos[0]
            self.axis = 1 # moves along y
        else:  # Bottom
            self.pos = np.array([self.np_random.uniform(0, self.screen_width), self.screen_height + margin], dtype=float)
            self.base_line = self.pos[0]
            self.axis = 1

        self.direction = 1 if side in [0, 2] else -1

    def update(self):
        self.trail.append(self.pos.copy())
        
        main_axis = self.axis
        perp_axis = 1 - self.axis
        
        # Linear motion
        self.pos[main_axis] += self.speed * self.direction
        
        # Sinusoidal motion
        if self.path_type == 'sin':
            offset = self.amplitude * math.sin(self.pos[main_axis] * self.frequency + self.phase)
        else: # cos
            offset = self.amplitude * math.cos(self.pos[main_axis] * self.frequency + self.phase)
        
        self.pos[perp_axis] = self.base_line + offset

    def get_rect(self):
        return pygame.Rect(self.pos[0] - self.size / 2, self.pos[1] - self.size / 2, self.size, self.size)

    def is_on_screen(self):
        margin = self.size * 3
        return -margin < self.pos[0] < self.screen_width + margin and \
               -margin < self.pos[1] < self.screen_height + margin

    def draw(self, surface, color):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        radius = int(self.size / 2)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)

    def draw_trail(self, surface, color):
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / len(self.trail)))
                trail_color = (*color, alpha)
                start_pos = (int(self.trail[i][0]), int(self.trail[i][1]))
                end_pos = (int(self.trail[i+1][0]), int(self.trail[i+1][1]))
                pygame.draw.line(surface, trail_color, start_pos, end_pos, 1)


class Particle:
    def __init__(self, pos, np_random):
        self.pos = np.array(pos, dtype=float)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.lifespan = np_random.integers(10, 20)
        self.age = 0
        self.size = np_random.uniform(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.age += 1
        return self.age < self.lifespan

    def draw(self, surface, color):
        alpha = max(0, 255 * (1 - self.age / self.lifespan))
        temp_color = (*color, int(alpha))
        
        # Use a temporary surface for alpha blending
        particle_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, temp_color, (self.size, self.size), self.size)
        surface.blit(particle_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))