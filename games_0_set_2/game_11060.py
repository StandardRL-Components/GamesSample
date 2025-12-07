import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:23:04.905784
# Source Brief: brief_01060.md
# Brief Index: 1060
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Building:
    """Helper class to store building state."""
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.has_artifact = False
        self.artifact_found = False

class Particle:
    """Helper class for visual effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # Gravity
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifespan))
            current_radius = int(self.radius * (self.life / self.lifespan))
            if current_radius > 0:
                # Using gfxdraw for anti-aliased circles
                color_with_alpha = self.color + (alpha,)
                pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), current_radius, color_with_alpha)
                pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), current_radius, color_with_alpha)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore a submerged city grid. Manipulate building heights to uncover all the hidden artifacts before you run out of steps."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the selector. Press space to grow a building and shift to shrink it. Uncover artifacts by shrinking buildings."
    )
    auto_advance = False

    # --- CONSTANTS ---
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT
    MAX_BUILDING_SIZE = 4
    MIN_BUILDING_SIZE = 1
    NUM_ARTIFACTS = 5
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_BUILDING_BASE = (20, 150, 170)
    COLOR_BUILDING_SHADOW = (10, 100, 120)
    COLOR_BUILDING_GLOW = (100, 220, 255)
    COLOR_ARTIFACT = (255, 215, 0)
    COLOR_SELECTION = (255, 255, 255)
    COLOR_UI_TEXT = (230, 230, 240)
    
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
        self.font_ui = pygame.font.Font(None, 32)
        
        self.grid = []
        self.particles = []
        self.selection = (0, 0)
        self.artifacts_found_count = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # This is a helper for development, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.artifacts_found_count = 0
        self.selection = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.particles.clear()

        # Initialize grid and buildings
        self.grid = [[Building(x, y, self.np_random.integers(self.MIN_BUILDING_SIZE, self.MAX_BUILDING_SIZE + 1))
                      for y in range(self.GRID_HEIGHT)] for x in range(self.GRID_WIDTH)]

        # Place artifacts
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        # Use self.np_random.choice for reproducibility with seeds
        artifact_indices = self.np_random.choice(len(all_coords), self.NUM_ARTIFACTS, replace=False)
        
        for idx in artifact_indices:
            x, y = all_coords[idx]
            self.grid[x][y].has_artifact = True
            self.grid[x][y].artifact_found = False
            # Ensure artifact is not revealed at start
            if self.grid[x][y].size == self.MIN_BUILDING_SIZE:
                self.grid[x][y].size = self.MIN_BUILDING_SIZE + 1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01 # Small cost for each step to encourage efficiency
        
        # --- 1. Handle Input and Actions ---
        sel_x, sel_y = self.selection
        if movement == 1: sel_y = (sel_y - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: sel_y = (sel_y + 1) % self.GRID_HEIGHT
        elif movement == 3: sel_x = (sel_x - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: sel_x = (sel_x + 1) % self.GRID_WIDTH
        self.selection = (sel_x, sel_y)

        selected_building = self.grid[sel_x][sel_y]

        if space_held and not shift_held: # Grow
            if selected_building.size < self.MAX_BUILDING_SIZE:
                selected_building.size += 1
                # SFX: Play a "grow" sound
                self._create_particles(self.selection, 30, 'grow')
                # Chain reaction: shrink neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = (sel_x + dx + self.GRID_WIDTH) % self.GRID_WIDTH, (sel_y + dy + self.GRID_HEIGHT) % self.GRID_HEIGHT
                    neighbor = self.grid[nx][ny]
                    if neighbor.size > self.MIN_BUILDING_SIZE:
                        neighbor.size -= 1
                        self._create_particles((nx, ny), 15, 'shrink')

        elif shift_held and not space_held: # Shrink
            if selected_building.size > self.MIN_BUILDING_SIZE:
                selected_building.size -= 1
                # SFX: Play a "shrink" sound
                self._create_particles(self.selection, 30, 'shrink')
                
                if selected_building.has_artifact and not selected_building.artifact_found and selected_building.size == self.MIN_BUILDING_SIZE:
                    selected_building.artifact_found = True
                    self.artifacts_found_count += 1
                    reward += 10.0 # Reward for finding an artifact
                    self.score += 10
                    # SFX: Play a "discovery" chime
                    self._create_particles(self.selection, 50, 'artifact')

        # --- 2. Update Game State ---
        self.steps += 1
        self._update_particles()
        
        # --- 3. Check for Termination ---
        terminated = False
        truncated = False
        if self.artifacts_found_count == self.NUM_ARTIFACTS:
            terminated = True
            reward += 100 # Victory reward
            self.score += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Game ends due to steps limit
            truncated = True # This is a truncation, not a failure state
            reward -= 10 # Penalty for timeout
            self.score -= 10
            
        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "artifacts_found": self.artifacts_found_count}

    def _render_game(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                building = self.grid[x][y]
                if building.artifact_found:
                    self._draw_artifact((x, y))

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self._draw_building(self.grid[x][y])

        self._draw_selection()
        
        for p in self.particles:
            p.draw(self.screen)

    def _draw_building(self, building):
        base_size = 10 + building.size * 6
        cx = (building.x + 0.5) * self.CELL_WIDTH
        cy = (building.y + 0.5) * self.CELL_HEIGHT
        
        shadow_offset = max(2, base_size * 0.1)
        shadow_rect = pygame.Rect(cx - base_size / 2, cy - base_size / 2, base_size, base_size)
        main_rect = pygame.Rect(cx - base_size / 2 - shadow_offset, cy - base_size / 2 - shadow_offset, base_size, base_size)

        glow_radius = int(base_size * 0.7)
        if glow_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(main_rect.centerx), int(main_rect.centery), glow_radius, self.COLOR_BUILDING_GLOW + (30,))
            pygame.gfxdraw.aacircle(self.screen, int(main_rect.centerx), int(main_rect.centery), glow_radius, self.COLOR_BUILDING_GLOW + (50,))

        pygame.draw.rect(self.screen, self.COLOR_BUILDING_SHADOW, shadow_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BUILDING_BASE, main_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BUILDING_GLOW, main_rect, width=1, border_radius=4)

    def _draw_artifact(self, pos):
        cx = (pos[0] + 0.5) * self.CELL_WIDTH
        cy = (pos[1] + 0.5) * self.CELL_HEIGHT
        
        shimmer = (math.sin(self.steps * 0.15) + 1) / 2
        radius = 12 + shimmer * 4
        alpha = 180 + shimmer * 75

        points = []
        for i in range(10):
            angle = math.radians(i * 36) + self.steps * 0.02
            r = radius if i % 2 == 0 else radius * 0.5
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

        color_with_alpha = self.COLOR_ARTIFACT + (int(alpha),)
        pygame.gfxdraw.aapolygon(self.screen, points, color_with_alpha)
        pygame.gfxdraw.filled_polygon(self.screen, points, color_with_alpha)

    def _draw_selection(self):
        sel_x, sel_y = self.selection
        rect = pygame.Rect(sel_x * self.CELL_WIDTH, sel_y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 150 + pulse * 105
        
        pygame.draw.rect(self.screen, self.COLOR_SELECTION + (int(alpha),), rect, width=3, border_radius=5)

    def _render_ui(self):
        text_str = f"Artifacts: {self.artifacts_found_count} / {self.NUM_ARTIFACTS}"
        text_surf = self.font_ui.render(text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        step_str = f"Steps: {self.steps} / {self.MAX_STEPS}"
        step_surf = self.font_ui.render(step_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(step_surf, (self.SCREEN_WIDTH - step_surf.get_width() - 10, 10))

    def _create_particles(self, grid_pos, count, p_type):
        cx = (grid_pos[0] + 0.5) * self.CELL_WIDTH
        cy = (grid_pos[1] + 0.5) * self.CELL_HEIGHT
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            
            if p_type == 'grow':
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                color = self.COLOR_BUILDING_GLOW
                radius = random.randint(3, 6)
                lifespan = random.randint(20, 40)
            elif p_type == 'shrink':
                vel = [math.cos(angle) * speed * -0.5, math.sin(angle) * speed * -0.5]
                color = self.COLOR_BUILDING_SHADOW
                radius = random.randint(2, 5)
                lifespan = random.randint(15, 30)
            elif p_type == 'artifact':
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                color = self.COLOR_ARTIFACT
                radius = random.randint(4, 8)
                lifespan = random.randint(30, 60)
            else:
                continue

            self.particles.append(Particle((cx, cy), vel, radius, color, lifespan))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for interactive testing and will not be run by the evaluation system.
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Atlantean Architect")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    
    print("Controls:\n  Arrows: Move selection\n  Space: Grow\n  Shift: Shrink\n  Q: Quit")

    last_action_time = pygame.time.get_ticks()
    ACTION_COOLDOWN = 150 # ms

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        current_time = pygame.time.get_ticks()
        action_taken = False
        keys = pygame.key.get_pressed()
        
        # Allow continuous key presses to register actions after a cooldown
        if current_time - last_action_time > ACTION_COOLDOWN:
            if keys[pygame.K_UP]: movement = 1; action_taken = True
            elif keys[pygame.K_DOWN]: movement = 2; action_taken = True
            elif keys[pygame.K_LEFT]: movement = 3; action_taken = True
            elif keys[pygame.K_RIGHT]: movement = 4; action_taken = True
            
            if keys[pygame.K_SPACE]: space = 1; action_taken = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1; action_taken = True

        if action_taken:
            last_action_time = current_time
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Found: {info['artifacts_found']}")
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                obs, info = env.reset()
                total_reward = 0
        else: # If no action, still need to update observation for animations
            obs = env._get_observation()

        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        
    env.close()