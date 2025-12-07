import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:26:14.224663
# Source Brief: brief_01010.md
# Brief Index: 1010
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Robot:
    """Represents a single robot unit."""
    def __init__(self, pos, target, size, generation, max_vel):
        self.pos = pygame.Vector2(pos)
        self.target = pygame.Vector2(target)
        self.size = float(size)
        self.generation = int(generation)
        self.replication_cooldown = 0
        self.trail = deque(maxlen=15)
        self.max_vel = max_vel

    def update(self, screen_width, screen_height):
        # Update trail
        if len(self.trail) == 0 or self.trail[-1] != (int(self.pos.x), int(self.pos.y)):
             self.trail.append((int(self.pos.x), int(self.pos.y)))

        # Move towards target
        dir_vec = self.target - self.pos
        dist = dir_vec.length()

        if dist > self.size / 2:
            # Velocity is inversely proportional to size (bigger = slower)
            velocity = self.max_vel / (1 + 0.1 * self.size)
            dir_vec.scale_to_length(min(dist, velocity))
            self.pos += dir_vec

        # Clamp position to screen bounds
        self.pos.x = max(self.size, min(self.pos.x, screen_width - self.size))
        self.pos.y = max(self.size, min(self.pos.y, screen_height - self.size))

        # Decrement cooldown
        if self.replication_cooldown > 0:
            self.replication_cooldown -= 1

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, size, lifetime, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = float(size)
        self.lifetime = int(lifetime)
        self.color = color

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)
        return self.lifetime > 0 and self.size > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cursor to command a growing swarm of robots. Replicate new units and transform "
        "existing ones to reach the target population before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to replicate a new robot "
        "and hold shift to transform the selected robot."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_DURATION_SECONDS = 60
    LOGIC_FPS = 60 # Steps per second
    MAX_STEPS = GAME_DURATION_SECONDS * LOGIC_FPS
    WIN_CONDITION_ROBOTS = 15
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_ROBOT_BASE = (0, 150, 255)
    COLOR_HIGHLIGHT = (255, 255, 255)
    COLOR_CURSOR = (200, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    
    # Gameplay Parameters
    INITIAL_ROBOT_SIZE = 10
    CURSOR_SPEED = 8
    ROBOT_MAX_VEL = 3.0
    REPLICATION_COOLDOWN = 60 # in steps (1 second)
    TRANSFORMATION_SIZE_INCREASE = 1.1 # 10% increase

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 60, bold=True)
        
        # State variables are initialized in reset()
        self.robots = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.robots = []
        self.particles = []
        
        # Create initial robot
        initial_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        initial_robot = Robot(
            pos=initial_pos,
            target=initial_pos,
            size=self.INITIAL_ROBOT_SIZE,
            generation=0,
            max_vel=self.ROBOT_MAX_VEL
        )
        self.robots.append(initial_robot)
        
        self.cursor_pos = pygame.Vector2(initial_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack and handle actions
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            reward += self._handle_input(movement, space_held, shift_held)
            
            # Update game state
            self._update_game_state()

        self.steps += 1
        
        # Check for termination conditions
        terminated = self._check_termination()
        
        # Calculate terminal rewards
        if terminated and not self.game_over:
            if len(self.robots) >= self.WIN_CONDITION_ROBOTS:
                reward += 100 # Victory bonus
                # sound: VICTORY
            else:
                reward += -100 # Timeout penalty
                # sound: TIMEOUT
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        """Processes player actions and returns immediate rewards."""
        reward = 0
        
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = max(0, min(self.cursor_pos.x, self.SCREEN_WIDTH))
        self.cursor_pos.y = max(0, min(self.cursor_pos.y, self.SCREEN_HEIGHT))
        
        # Find robot closest to cursor for actions
        source_robot = self._find_closest_robot_to_cursor()
        if source_robot is None:
            return 0

        # Action: Replicate (Space)
        if space_held and source_robot.replication_cooldown == 0 and len(self.robots) < self.WIN_CONDITION_ROBOTS:
            new_robot = Robot(
                pos=source_robot.pos,
                target=self.cursor_pos,
                size=self.INITIAL_ROBOT_SIZE,
                generation=source_robot.generation + 1,
                max_vel=source_robot.max_vel * 0.8 # Inherits speed with a penalty
            )
            self.robots.append(new_robot)
            source_robot.replication_cooldown = self.REPLICATION_COOLDOWN
            self._create_particles(source_robot.pos, self._get_robot_color(source_robot.generation), 20, is_burst=True)
            # sound: REPLICATE
            reward += 1.0 # Reward for creating a new robot

        # Action: Transform (Shift)
        if shift_held:
            source_robot.size *= self.TRANSFORMATION_SIZE_INCREASE
            self._create_particles(source_robot.pos, self.COLOR_HIGHLIGHT, 5, is_burst=False)
            # sound: TRANSFORM
            # No direct reward for transformation, its value is strategic

        return reward

    def _update_game_state(self):
        """Updates all robots and particles."""
        # Update robots
        for robot in self.robots:
            robot.update(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

    def _check_termination(self):
        """Checks if the episode should end."""
        win = len(self.robots) >= self.WIN_CONDITION_ROBOTS
        timeout = self.steps >= self.MAX_STEPS
        return win or timeout

    def _find_closest_robot_to_cursor(self):
        """Finds the robot nearest to the on-screen cursor."""
        if not self.robots:
            return None
        
        closest_robot = min(
            self.robots, 
            key=lambda r: self.cursor_pos.distance_squared_to(r.pos)
        )
        return closest_robot

    def _create_particles(self, pos, color, count, is_burst=False):
        """Generates particles for a visual effect."""
        for _ in range(count):
            if is_burst:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            else: # Implosion/glow effect
                vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)) * 0.5
            
            size = random.uniform(2, 5)
            lifetime = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, size, lifetime, color))

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_effects()
        self._render_robots()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robots": len(self.robots),
            "time_left": (self.MAX_STEPS - self.steps) / self.LOGIC_FPS,
        }

    def _get_robot_color(self, generation):
        """Calculates robot color based on its generation."""
        r, g, b = self.COLOR_ROBOT_BASE
        # Get brighter with each generation
        g = min(255, g + generation * 15)
        b = min(255, b + generation * 5)
        return (r, g, b)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_effects(self):
        # Render particles
        for p in self.particles:
            pos = (int(p.pos.x), int(p.pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p.size), p.color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p.size), p.color)
            
    def _render_robots(self):
        closest_robot = self._find_closest_robot_to_cursor()

        for robot in sorted(self.robots, key=lambda r: r.size): # Draw smaller ones last
            # Render trail
            if len(robot.trail) > 1:
                color = self._get_robot_color(robot.generation)
                alpha_color = (color[0], color[1], color[2], 100)
                pygame.draw.lines(self.screen, color, False, list(robot.trail), width=int(robot.size/4))

            # Render robot body
            pos = (int(robot.pos.x), int(robot.pos.y))
            color = self._get_robot_color(robot.generation)
            radius = int(robot.size)
            
            # Glow effect
            glow_radius = int(radius * 1.5)
            glow_color = (color[0], color[1], color[2], 100) # Semi-transparent
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

            # Render highlight for closest robot
            if robot is closest_robot:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_HIGHLIGHT)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 4, self.COLOR_HIGHLIGHT)

    def _render_ui(self):
        # Render cursor
        c_pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        size = 8
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (c_pos[0] - size, c_pos[1]), (c_pos[0] + size, c_pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (c_pos[0], c_pos[1] - size), (c_pos[0], c_pos[1] + size), 2)

        # Render robot count
        robot_text = self.font_ui.render(f"ROBOTS: {len(self.robots)}/{self.WIN_CONDITION_ROBOTS}", True, self.COLOR_TEXT)
        self.screen.blit(robot_text, (10, 10))
        
        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.LOGIC_FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Render game over message
        if self.game_over:
            if len(self.robots) >= self.WIN_CONDITION_ROBOTS:
                msg = "VICTORY"
                color = (100, 255, 100)
            else:
                msg = "TIME UP"
                color = (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for testing
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local development.
    # It requires a display, so we unset the dummy videodriver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Swarm")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                
        clock.tick(env.LOGIC_FPS)

    print(f"Game Over. Final Info: {info}")
    env.close()