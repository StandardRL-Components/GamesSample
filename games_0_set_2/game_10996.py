import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:16:39.001047
# Source Brief: brief_00996.md
# Brief Index: 996
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class LineSegment:
    """Represents a single segment of the player's line."""
    def __init__(self, x, y, width, screen_width):
        self.x = x
        self.y = y
        self.width = width
        self.screen_width = screen_width
        self.state = 'straight'  # 'straight' or 'curved'
        self.vx = 0
        self.thickness = 6

    def get_rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.thickness / 2, self.width, self.thickness)

    def update(self, dt, horizontal_accel, state_change=None):
        # Update state if changed
        if state_change:
            self.state = state_change

        # Update horizontal velocity and position
        self.vx += horizontal_accel * dt
        self.vx *= 0.9  # Damping
        self.x += self.vx * dt
        
        # Clamp to screen bounds
        half_width = self.width / 2
        self.x = max(half_width, min(self.screen_width - half_width, self.x))
        if self.x == half_width or self.x == self.screen_width - half_width:
            self.vx = 0

        # Update vertical position based on state
        speed = GameEnv.LINE_SPEED_FAST if self.state == 'straight' else GameEnv.LINE_SPEED_SLOW
        self.y -= speed * dt

    def draw(self, surface):
        color = GameEnv.COLOR_LINE
        glow_color = GameEnv.COLOR_LINE_GLOW
        
        # Draw glow effect
        for i in range(4, 0, -1):
            glow_alpha = 40 - i * 8
            glow_surface = pygame.Surface((self.width + i * 4, self.thickness + i * 4), pygame.SRCALPHA)
            if self.state == 'straight':
                pygame.draw.rect(glow_surface, (*glow_color, glow_alpha), glow_surface.get_rect(), border_radius=3)
            else:
                arc_rect = pygame.Rect(0, 0, self.width + i * 4, 40 + i*4)
                pygame.draw.arc(glow_surface, (*glow_color, glow_alpha), arc_rect, math.pi, math.pi * 2, self.thickness + i*2)
            
            draw_pos = (int(self.x - (self.width / 2) - i * 2), int(self.y - (self.thickness / 2) - i * 2) if self.state == 'straight' else int(self.y - 20 - i*2))
            surface.blit(glow_surface, draw_pos)


        # Draw main line
        if self.state == 'straight':
            rect = pygame.Rect(0, 0, self.width, self.thickness)
            rect.center = (self.x, self.y)
            pygame.draw.rect(surface, color, rect, border_radius=3)
        else:
            # Curved line (arc)
            arc_rect = pygame.Rect(self.x - self.width / 2, self.y - 20, self.width, 40)
            pygame.draw.arc(surface, color, arc_rect, math.pi, math.pi * 2, self.thickness)

class Particle:
    """Represents a single particle for effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.lifespan = random.randint(20, 40)
        self.color = color
        self.radius = self.lifespan / 6

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius = max(0, self.lifespan / 6)

    def draw(self, surface):
        if self.lifespan > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), self.color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a morphing line through a field of obstacles to reach the finish line. "
        "Change your line's shape to control its speed and maneuver carefully."
    )
    user_guide = (
        "Controls: Use ←→ to move left and right. Hold space to morph into a slow, curved line. "
        "Hold shift to morph into a fast, straight line."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 60, 100)
    COLOR_LINE = (255, 255, 255)
    COLOR_LINE_GLOW = (180, 220, 255)
    COLOR_OBSTACLE = (220, 50, 50)
    COLOR_OBSTACLE_GLOW = (150, 20, 20)
    COLOR_FINISH = (50, 220, 50)
    COLOR_FINISH_GLOW = (20, 150, 20)
    COLOR_UI_TEXT = (230, 230, 230)

    # Game Mechanics
    LINE_START_Y = 380
    LINE_SPEED_FAST = 100  # pixels per second
    LINE_SPEED_SLOW = 40   # pixels per second
    LINE_HORIZONTAL_ACCEL = 250 # pixels per second^2
    MIN_LINE_WIDTH = 20
    FINISH_LINE_Y = 20
    MAX_LINES = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.dt = 1 / self.FPS

        # State variables are initialized in reset()
        self.lines = []
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_spawn_timer = 0
        self.base_obstacle_count = 5

        # self.reset() is not called here to avoid creating a surface before it might be needed
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_spawn_timer = 0
        self.base_obstacle_count = 5

        # Initialize player line
        self.lines = [LineSegment(self.SCREEN_WIDTH / 2, self.LINE_START_Y, self.SCREEN_WIDTH * 0.8, self.SCREEN_WIDTH)]
        
        # Initialize obstacles
        self.obstacles = []
        self._generate_obstacles(self.base_obstacle_count)

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        self._update_physics()
        
        # --- Handle Collisions and Rewards ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check for Termination and Win/Loss Rewards ---
        terminated, win_reward = self._check_termination()
        reward += win_reward
        self.score += reward
        self.game_over = terminated

        # --- Spawn new obstacles periodically ---
        self.obstacle_spawn_timer += 1
        if self.obstacle_spawn_timer >= 200:
            self.obstacle_spawn_timer = 0
            self.base_obstacle_count = int(self.base_obstacle_count * 1.05) + 1
            self._generate_obstacles(1)
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        horizontal_accel = 0
        if movement == 3: # Left
            horizontal_accel = -self.LINE_HORIZONTAL_ACCEL
        elif movement == 4: # Right
            horizontal_accel = self.LINE_HORIZONTAL_ACCEL
        
        state_change = None
        if space_held: # Morph to curved (slow)
            state_change = 'curved'
        elif shift_held: # Morph to straight (fast)
            state_change = 'straight'
        
        for line in self.lines:
            line.update(self.dt, horizontal_accel, state_change)

    def _update_physics(self):
        # Update particles and remove dead ones
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _handle_collisions(self):
        collision_reward = 0
        new_lines = []
        lines_to_remove = []

        for i, line in enumerate(self.lines):
            line_rect = line.get_rect()
            collided = False
            for obstacle in self.obstacles:
                if line_rect.colliderect(obstacle):
                    collided = True
                    break
            
            if collided:
                # SFX: Play collision sound
                lines_to_remove.append(line)
                collision_reward -= 5.0
                
                # Create particle explosion
                for _ in range(30):
                    self.particles.append(Particle(line.x, line.y, self.COLOR_OBSTACLE))

                # Split the line if it's wide enough
                if line.width > self.MIN_LINE_WIDTH * 2 and len(self.lines) + len(new_lines) < self.MAX_LINES:
                    new_width = line.width / 2 - 10
                    new_lines.append(LineSegment(line.x - line.width / 4, line.y, new_width, self.SCREEN_WIDTH))
                    new_lines.append(LineSegment(line.x + line.width / 4, line.y, new_width, self.SCREEN_WIDTH))
        
        if lines_to_remove:
            self.lines = [line for line in self.lines if line not in lines_to_remove]
            self.lines.extend(new_lines)
            
        return collision_reward

    def _check_termination(self):
        terminated = False
        win_reward = 0

        # Win condition
        for line in self.lines:
            if line.y <= self.FINISH_LINE_Y:
                terminated = True
                win_reward = 100.0
                # SFX: Play win sound
                break
        
        # Loss condition
        if not self.lines and not terminated:
            terminated = True
            # SFX: Play lose sound
        
        return terminated, win_reward
        
    def _generate_obstacles(self, count):
        for _ in range(count):
            while True:
                width = random.randint(40, 120)
                height = random.randint(15, 30)
                x = random.randint(0, self.SCREEN_WIDTH - width)
                y = random.randint(self.FINISH_LINE_Y + 20, self.LINE_START_Y - 50)
                new_obstacle = pygame.Rect(x, y, width, height)
                
                # Ensure no overlap with existing obstacles
                if not any(new_obstacle.colliderect(obs) for obs in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break

    def _get_observation(self):
        # --- Draw Background Gradient ---
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # --- Render Game Elements ---
        self._render_game()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw finish line
        finish_rect = pygame.Rect(0, self.FINISH_LINE_Y, self.SCREEN_WIDTH, 5)
        pygame.draw.rect(self.screen, self.COLOR_FINISH_GLOW, finish_rect.inflate(0, 8))
        pygame.draw.rect(self.screen, self.COLOR_FINISH, finish_rect)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, obs.inflate(8, 8), border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=5)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw lines
        for line in self.lines:
            line.draw(self.screen)
            
    def _render_ui(self):
        # Segments count
        segments_text = self.font.render(f"SEGMENTS: {len(self.lines)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(segments_text, (10, 10))
        
        # Score
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "segments": len(self.lines)
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a display for manual playing
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Morph Line")
    clock = pygame.time.Clock()

    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            obs, info = env.reset()
            terminated = False

        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Display the observation ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()