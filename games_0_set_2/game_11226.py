import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:48:39.450246
# Source Brief: brief_01226.md
# Brief Index: 1226
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

    game_description = (
        "Repair damaged systems in a derelict maze. Navigate to broken components and use the correct tool before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move, press Space to cycle tools, and press Shift to use the selected tool."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    MAX_STEPS = 1000

    COLOR_BG = (15, 20, 35)
    COLOR_WALL = (40, 50, 70)
    COLOR_FLOOR = (25, 30, 45)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_REPAIRED = (0, 255, 100)
    COLOR_TEXT = (220, 220, 240)

    TOOLS = ["Wrench", "Welder", "Coolant"]
    TOOL_COLORS = {
        "Wrench": (255, 180, 0),
        "Welder": (255, 100, 0),
        "Coolant": (0, 200, 200)
    }
    DAMAGE_COLORS = {
        "Wrench": (200, 50, 50),
        "Welder": (255, 0, 0),
        "Coolant": (230, 0, 230)
    }

    class Particle:
        def __init__(self, x, y, color):
            self.x = x
            self.y = y
            self.color = color
            self.vx = random.uniform(-1.5, 1.5)
            self.vy = random.uniform(-1.5, 1.5)
            self.life = random.randint(20, 40)
            self.radius = random.uniform(2, 5)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.life -= 1
            self.radius -= 0.1
            return self.life > 0 and self.radius > 0

        def draw(self, surface):
            if self.life > 0 and self.radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        self.grid = []
        self.components = []
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.selected_tool_idx = 0
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # The reset method will be called to initialize the state
        # self.reset() is removed from here to follow gymnasium standard practice
        
        # self.validate_implementation() # This can be removed from production code

    def _generate_level(self):
        self.grid = [[1 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        # Randomized DFS for maze generation
        stack = [(1, 1)]
        self.grid[1][1] = 0
        visited = set([(1, 1)])

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH -1 and 0 < ny < self.GRID_HEIGHT -1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid[ny][nx] = 0
                self.grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        open_cells = [(x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) if self.grid[y][x] == 0]
        
        # Player start position
        start_pos_idx = self.np_random.integers(0, len(open_cells))
        self.player_pos = list(open_cells.pop(start_pos_idx))
        self.player_visual_pos = [float(self.player_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                                  float(self.player_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)]

        # Damaged components
        self.components = []
        num_components = 3
        component_indices = self.np_random.choice(len(open_cells), size=min(num_components, len(open_cells)), replace=False)
        component_cells = [open_cells[i] for i in component_indices]

        for x, y in component_cells:
            self.components.append({
                "pos": [x, y],
                "tool_needed": random.choice(self.TOOLS), # using random for now as it's part of game logic
                "repaired": False
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_tool_idx = 0
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0
        
        # --- Action Handling ---
        # Detect button presses (rising edge)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # 1. Movement
        dist_before = self._get_dist_to_closest_unrepaired()
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT and self.grid[new_y][new_x] == 0:
                self.player_pos = [new_x, new_y]
        
        dist_after = self._get_dist_to_closest_unrepaired()
        if dist_after is not None and dist_before is not None and dist_after < dist_before:
            reward += 0.1 # Reward for moving closer

        # 2. Cycle Tool (Space)
        if space_pressed:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.TOOLS)
            # Reward for selecting correct tool for nearest component
            closest_comp = self._get_closest_unrepaired_component()
            if closest_comp and self.TOOLS[self.selected_tool_idx] == closest_comp["tool_needed"]:
                reward += 1.0

        # 3. Use Tool (Shift)
        if shift_pressed:
            for comp in self.components:
                if not comp["repaired"] and comp["pos"] == self.player_pos:
                    if self.TOOLS[self.selected_tool_idx] == comp["tool_needed"]:
                        comp["repaired"] = True
                        self.score += 1
                        reward += 5.0
                        # Particle burst on repair
                        px, py = self._grid_to_pixel(self.player_pos)
                        for _ in range(50):
                            self.particles.append(self.Particle(px, py, self.COLOR_REPAIRED))
                    else:
                        pass # Optional penalty for wrong tool

        # --- Update Game State ---
        self.steps += 1
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Update player visual position for smooth movement
        target_vx = self.player_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        target_vy = self.player_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player_visual_pos[0] += (target_vx - self.player_visual_pos[0]) * 0.5
        self.player_visual_pos[1] += (target_vy - self.player_visual_pos[1]) * 0.5

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and all(c["repaired"] for c in self.components):
            reward += 100.0 # Victory bonus
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_dist_to_closest_unrepaired(self):
        unrepaired_comps = [c for c in self.components if not c["repaired"]]
        if not unrepaired_comps:
            return None
        
        player_p = np.array(self.player_pos)
        dists = [np.linalg.norm(player_p - np.array(c["pos"])) for c in unrepaired_comps]
        return min(dists)
        
    def _get_closest_unrepaired_component(self):
        unrepaired_comps = [c for c in self.components if not c["repaired"]]
        if not unrepaired_comps:
            return None
        
        player_p = np.array(self.player_pos)
        dists = [np.linalg.norm(player_p - np.array(c["pos"])) for c in unrepaired_comps]
        return unrepaired_comps[np.argmin(dists)]

    def _check_termination(self):
        all_repaired = all(c["repaired"] for c in self.components)
        return all_repaired

    def _grid_to_pixel(self, grid_pos):
        return (grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, 
                grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = self.COLOR_WALL if self.grid[y][x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw components
        for comp in self.components:
            px, py = self._grid_to_pixel(comp["pos"])
            if comp["repaired"]:
                pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 3, self.COLOR_REPAIRED)
                pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 3, self.COLOR_REPAIRED)
            else:
                # Flashing effect for damaged components
                flash_alpha = (math.sin(self.steps * 0.2) + 1) / 2
                color = self.DAMAGE_COLORS[comp["tool_needed"]]
                flash_color = tuple(int(c * flash_alpha + self.COLOR_FLOOR[i] * (1-flash_alpha)) for i, c in enumerate(color))
                pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 3, flash_color)
                pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 3, color)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw player
        px, py = int(self.player_visual_pos[0]), int(self.player_visual_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, px, py, 10, self.COLOR_PLAYER_GLOW)
        # Player core
        pygame.gfxdraw.filled_circle(self.screen, px, py, 6, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, 6, self.COLOR_TEXT)
        
    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((20, 25, 40, 180))
        self.screen.blit(ui_panel, (0, 0))

        # Repairs
        repaired_count = sum(1 for c in self.components if c["repaired"])
        total_count = len(self.components)
        repair_text = f"SYSTEMS: {repaired_count}/{total_count}"
        text_surf = self.font_ui.render(repair_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Steps
        steps_text = f"TIME: {self.MAX_STEPS - self.steps}"
        text_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))

        # Selected Tool
        tool_name = self.TOOLS[self.selected_tool_idx]
        tool_color = self.TOOL_COLORS[tool_name]
        tool_text = f"TOOL: {tool_name}"
        text_surf = self.font_ui.render(tool_text, True, tool_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "MISSION COMPLETE" if all(c["repaired"] for c in self.components) else "MISSION FAILED"
            color = self.COLOR_REPAIRED if all(c["repaired"] for c in self.components) else self.DAMAGE_COLORS["Welder"]
            text_surf = self.font_big.render(msg, True, color)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surf.get_height() // 2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "repaired_components": sum(1 for c in self.components if c["repaired"]),
            "total_components": len(self.components),
            "player_pos": self.player_pos,
            "selected_tool": self.TOOLS[self.selected_tool_idx]
        }
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example usage: run the environment with a random agent
    # This part requires a graphical display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use pygame for human interaction
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Celestial Surgeon")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("  - R: Reset")

    while True:
        # Human input
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the screen
        # Need to transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS