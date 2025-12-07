
# Generated: 2025-08-28T06:44:44.666912
# Source Brief: brief_03027.md
# Brief Index: 3027

        
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


class Laser:
    """Represents a single laser beam."""
    def __init__(self, rect, axis, move_range, speed, color, glow_color, np_random):
        self.start_rect = rect.copy()
        self.rect = rect.copy()
        self.axis = axis
        self.move_range = move_range
        self.speed = speed
        self.initial_speed = speed
        self.direction = np_random.choice([-1, 1])
        self.color = color
        self.glow_color = glow_color

    def update(self, speed_increase_factor):
        self.speed += speed_increase_factor
        if self.axis == 'x':
            self.rect.x += self.speed * self.direction
            if self.rect.right > self.move_range[1] or self.rect.left < self.move_range[0]:
                self.direction *= -1
                self.rect.x = np.clip(self.rect.x, self.move_range[0], self.move_range[1] - self.rect.width)
        else:
            self.rect.y += self.speed * self.direction
            if self.rect.bottom > self.move_range[1] or self.rect.top < self.move_range[0]:
                self.direction *= -1
                self.rect.y = np.clip(self.rect.y, self.move_range[0], self.move_range[1] - self.rect.height)

    def draw(self, surface):
        glow_rect = self.rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.glow_color, glow_surf.get_rect(), border_radius=5)
        surface.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)

    def reset(self):
        self.rect = self.start_rect.copy()
        self.speed = self.initial_speed
        self.direction = 1

class Particle:
    """Represents a single particle for effects."""
    def __init__(self, pos, np_random):
        self.pos = list(pos)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 5)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.life = np_random.integers(25, 50)
        self.initial_life = self.life
        self.color = (180, 255, 180) # Light green/white

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.initial_life))
        radius = int(3 * (self.life / self.initial_life))
        if radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), radius, self.color + (alpha,))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the robot. Avoid the red lasers and reach the green exit."
    )

    game_description = (
        "Guide a robot through laser-filled corridors to the exit. Plan your path carefully as the lasers sweep across the level. Each move counts!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALL = (40, 45, 65)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_ACCENT = (100, 200, 255)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_LASER_GLOW = (255, 0, 0, 40)
        self.COLOR_UI = (220, 220, 240)
        
        # Game state variables
        self.robot_pos = None
        self.exit_pos = None
        self.walls = []
        self.lasers = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self._create_level()
        
        return self._get_observation(), self._get_info()

    def _create_level(self):
        """Defines the static layout of the level: walls, robot start, exit, and lasers."""
        self.robot_pos = np.array([2, 2])
        self.exit_pos = np.array([self.GRID_W - 3, self.GRID_H - 3])

        wall_coords = set()
        # Borders
        for x in range(self.GRID_W):
            wall_coords.add((x, 0))
            wall_coords.add((x, self.GRID_H - 1))
        for y in range(self.GRID_H):
            wall_coords.add((0, y))
            wall_coords.add((self.GRID_W - 1, y))

        # Internal walls for a simple maze
        for y in range(5, self.GRID_H - 5):
            wall_coords.add((self.GRID_W // 3, y))
        for y in range(0, self.GRID_H - 8):
            wall_coords.add((2 * self.GRID_W // 3, y))

        self.walls = [pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE) for x, y in wall_coords]

        self.exit_rect = pygame.Rect(self.exit_pos[0] * self.GRID_SIZE, self.exit_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)

        # Initialize lasers
        self.lasers = []
        # Vertical laser
        laser1_rect = pygame.Rect((self.GRID_W // 3 + 4) * self.GRID_SIZE, 1 * self.GRID_SIZE, 8, self.GRID_SIZE * 3)
        self.lasers.append(Laser(laser1_rect, 'y', (1 * self.GRID_SIZE, (self.GRID_H - 4) * self.GRID_SIZE), 2, self.COLOR_LASER, self.COLOR_LASER_GLOW, self.np_random))
        
        # Horizontal laser
        laser2_rect = pygame.Rect(1 * self.GRID_SIZE, (self.GRID_H - 6) * self.GRID_SIZE, self.GRID_SIZE * 4, 8)
        self.lasers.append(Laser(laser2_rect, 'x', (1 * self.GRID_SIZE, (2 * self.GRID_W // 3 - 5) * self.GRID_SIZE), 2.5, self.COLOR_LASER, self.COLOR_LASER_GLOW, self.np_random))

        for laser in self.lasers:
            laser.reset()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # Update game logic
        self.steps += 1
        self._move_robot(movement)
        self._update_lasers()
        self._update_particles()
        
        # Check for termination and calculate reward
        reward, terminated = self._check_game_state()
        self.score += reward
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_robot(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        target_pos = self.robot_pos.copy()
        if movement == 1: target_pos[1] -= 1
        elif movement == 2: target_pos[1] += 1
        elif movement == 3: target_pos[0] -= 1
        elif movement == 4: target_pos[0] += 1
        
        if movement != 0:
            target_rect = pygame.Rect(target_pos[0] * self.GRID_SIZE, target_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            if target_rect.collidelist(self.walls) == -1:
                self.robot_pos = target_pos

    def _update_lasers(self):
        # Speed increases after 50 steps, as per brief
        speed_increase = 0.05 if self.steps > 50 else 0
        for laser in self.lasers:
            laser.update(speed_increase)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _check_game_state(self):
        robot_rect = pygame.Rect(self.robot_pos[0] * self.GRID_SIZE, self.robot_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        # Check laser collision
        for laser in self.lasers:
            if robot_rect.colliderect(laser.rect):
                # sfx: player_zap
                return -10.0, True

        # Check exit collision
        if robot_rect.colliderect(self.exit_rect):
            self._create_victory_particles()
            # sfx: victory_fanfare
            return 10.0, True

        # Check max steps
        if self.steps >= 1000:
            return 0.0, True
            
        # Standard step cost
        return -0.1, False

    def _create_victory_particles(self):
        for _ in range(50):
            self.particles.append(Particle(self.exit_rect.center, self.np_random))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Draw lasers
        for laser in self.lasers:
            laser.draw(self.screen)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw robot
        robot_rect = pygame.Rect(self.robot_pos[0] * self.GRID_SIZE, self.robot_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_ACCENT, robot_rect.inflate(-6, -6), border_radius=3)

    def _render_ui(self):
        steps_text = self.font.render(f"STEPS: {self.steps}", True, self.COLOR_UI)
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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