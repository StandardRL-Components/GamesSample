import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:21:07.751975
# Source Brief: brief_01672.md
# Brief Index: 1672
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes for game objects
class Component:
    def __init__(self, x, y, comp_type, grid_pos):
        self.x, self.y = x, y
        self.type = comp_type
        self.grid_pos = grid_pos
        self.is_active = False
        self.animation_timer = random.uniform(0, 1)
        self.connections = {'N': None, 'S': None, 'E': None, 'W': None}
        self.is_powered = False

class Particle:
    def __init__(self, x, y, p_type, path):
        self.x, self.y = x, y
        self.type = p_type
        self.path = deque(path)
        self.target_pos = self.path.popleft() if self.path else (x, y)
        self.progress = 0.0
        self.speed = 2.0
        self.trail = deque(maxlen=10)

class FlyingComponent:
    def __init__(self, start_pos, end_pos, comp_type, grid_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.current_pos = start_pos
        self.type = comp_type
        self.grid_pos = grid_pos
        self.progress = 0.0
        self.speed = 0.05 # Progress per frame

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a factory on a fractal landscape. Place components to create a production line, "
        "process raw materials, and maximize your throughput."
    )
    user_guide = (
        "Controls: Use arrow keys to move the aimer. Press space to launch the selected component. "
        "Hold shift to slow down time."
    )
    auto_advance = True

    # --- INITIALIZATION ---
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
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 40)

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_FRACTAL_1 = (25, 30, 45)
        self.COLOR_FRACTAL_2 = (35, 42, 63)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_AIMER = (255, 255, 255)
        self.COLOR_TRAJECTORY = (255, 255, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_SHADOW = (10, 10, 15)
        self.COLOR_TIMESLOW_OVERLAY = (50, 80, 200, 50)

        self.RESOURCE_COLORS = {
            "RAW": (255, 80, 80),      # Red
            "ENERGY": (100, 255, 100), # Green
            "WATER": (80, 150, 255),   # Blue
            "PROCESSED": (255, 220, 80) # Yellow
        }
        self.COMPONENT_TYPES = ["CONVEYOR", "PROCESSOR", "PORTAL"]
        self.COMPONENT_COLORS = {
            "CONVEYOR": (150, 150, 170),
            "PROCESSOR": (220, 100, 220),
            "PORTAL": (100, 220, 220),
            "SOURCE": (255, 255, 255),
            "SINK": (50, 50, 50)
        }
        
        # Pre-render fractal background for performance
        self.fractal_surface = self._create_fractal_surface()

        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    # --- GYMNASIUM INTERFACE METHODS ---
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            random.seed(seed)
        else:
            super().reset()


        # Game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.max_throughput_achieved = 0.0
        self.current_throughput = 0.0
        self.throughput_history = deque(maxlen=60) # Track last 2 seconds of production

        # Player/Aimer state
        self.aimer_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.launcher_pos = (self.GRID_SIZE // 2, self.HEIGHT // 2)

        # Game objects
        self.components = {} # Keyed by (gx, gy) tuple
        self.flying_components = []
        self.particles = []

        # Time and action state
        self.time_slow_active = False
        self.prev_space_held = False
        self.selected_component_idx = 0
        
        # Place initial source and sink
        self._place_initial_components()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        # --- 1. HANDLE INPUT ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.time_slow_active = shift_held

        # Move aimer
        if movement == 1: self.aimer_grid_pos[1] -= 1 # Up
        if movement == 2: self.aimer_grid_pos[1] += 1 # Down
        if movement == 3: self.aimer_grid_pos[0] -= 1 # Left
        if movement == 4: self.aimer_grid_pos[0] += 1 # Right
        self.aimer_grid_pos[0] = np.clip(self.aimer_grid_pos[0], 0, self.GRID_W - 1)
        self.aimer_grid_pos[1] = np.clip(self.aimer_grid_pos[1], 0, self.GRID_H - 1)

        # Launch component on space press (rising edge)
        if space_held and not self.prev_space_held:
            reward += self._launch_component()
        self.prev_space_held = space_held

        # --- 2. UPDATE GAME STATE ---
        time_factor = 0.2 if self.time_slow_active else 1.0
        
        # Update flying components
        reward += self._update_flying_components(time_factor)
        
        # Update particles
        processed_count = self._update_particles(time_factor)
        reward += processed_count * 0.1
        self.throughput_history.append(processed_count)

        # Spawn new particles
        if self.steps % 10 == 0:
            self._spawn_particles()
        
        # --- 3. CALCULATE REWARDS & TERMINATION ---
        self.current_throughput = sum(self.throughput_history) * 3 # Extrapolate to per-minute
        if self.current_throughput > self.max_throughput_achieved * 1.1 and self.max_throughput_achieved > 0:
            reward += 10.0
            self.max_throughput_achieved = self.current_throughput
        elif self.current_throughput > self.max_throughput_achieved:
             self.max_throughput_achieved = self.current_throughput

        self.score += reward
        terminated = self.steps >= 1000

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def close(self):
        pygame.quit()

    # --- PRIVATE HELPER METHODS ---
    def _get_observation(self):
        # Main render call
        self.screen.blit(self.fractal_surface, (0, 0))
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "throughput": self.current_throughput,
        }

    def _place_initial_components(self):
        source_pos = (2, self.GRID_H // 2)
        sink_pos = (self.GRID_W - 3, self.GRID_H // 2)
        
        self.components[source_pos] = Component(
            source_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
            source_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2,
            "SOURCE", source_pos)
        self.components[source_pos].is_powered = True

        self.components[sink_pos] = Component(
            sink_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
            sink_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2,
            "SINK", sink_pos)
        self._update_power_grid()

    def _launch_component(self):
        grid_pos = tuple(self.aimer_grid_pos)
        
        # Prevent building on existing components or invalid fractal areas
        if grid_pos in self.components or not self._is_buildable(grid_pos):
            # Sound: Negative beep
            return -0.1 # Small penalty for invalid action

        comp_type = self.COMPONENT_TYPES[self.selected_component_idx]
        target_center = (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, 
                         grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
        
        self.flying_components.append(
            FlyingComponent(self.launcher_pos, target_center, comp_type, grid_pos)
        )
        # Sound: Launch swoosh
        
        # Cycle to next component type
        self.selected_component_idx = (self.selected_component_idx + 1) % len(self.COMPONENT_TYPES)
        return 0.0

    def _update_flying_components(self, time_factor):
        reward = 0.0
        for fc in self.flying_components[:]:
            fc.progress += fc.speed * time_factor
            fc.current_pos = (
                self.launcher_pos[0] + (fc.end_pos[0] - self.launcher_pos[0]) * fc.progress,
                self.launcher_pos[1] + (fc.end_pos[1] - self.launcher_pos[1]) * fc.progress
            )
            if fc.progress >= 1.0:
                self.flying_components.remove(fc)
                # Sound: Component lands with a thud
                reward += self._place_component(fc)
        return reward

    def _place_component(self, flying_comp):
        grid_pos = flying_comp.grid_pos
        if grid_pos in self.components or not self._is_buildable(grid_pos):
            return -1.0 # Penalty for landing on invalid spot

        new_comp = Component(flying_comp.end_pos[0], flying_comp.end_pos[1], flying_comp.type, grid_pos)
        self.components[grid_pos] = new_comp
        
        # Check for new connections and update power
        reward = self._update_power_grid()
        return reward

    def _update_power_grid(self):
        """Calculates which components are connected to the source."""
        # Reset power status
        for comp in self.components.values():
            comp.is_powered = (comp.type == "SOURCE")

        # Propagate power through the grid using a queue
        q = deque([c for c in self.components.values() if c.is_powered])
        visited = set(c.grid_pos for c in q)
        
        connection_reward = 0

        while q:
            current_comp = q.popleft()
            gx, gy = current_comp.grid_pos
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_pos = (gx + dx, gy + dy)
                if neighbor_pos in self.components and neighbor_pos not in visited:
                    neighbor_comp = self.components[neighbor_pos]
                    if neighbor_comp.type != "SINK":
                        neighbor_comp.is_powered = True
                        visited.add(neighbor_pos)
                        q.append(neighbor_comp)
                        connection_reward += 1.0 # Reward for making a new connection
        return connection_reward

    def _spawn_particles(self):
        for comp in self.components.values():
            if comp.type == "SOURCE":
                path = self._find_path_to_sink(comp.grid_pos)
                if path:
                    # Sound: Particle spawn blip
                    self.particles.append(Particle(comp.x, comp.y, "RAW", path))

    def _find_path_to_sink(self, start_grid_pos):
        q = deque([(start_grid_pos, [self.components[start_grid_pos].grid_pos])])
        visited = {start_grid_pos}

        while q:
            (vx, vy), path = q.popleft()
            
            # Check for sink
            if self.components[(vx, vy)].type == "SINK":
                return [self.components[p].__dict__ for p in path]

            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_pos = (vx + dx, vy + dy)
                if neighbor_pos in self.components and neighbor_pos not in visited:
                    if self.components[neighbor_pos].is_powered or self.components[neighbor_pos].type == "SINK":
                        new_path = list(path)
                        new_path.append(neighbor_pos)
                        visited.add(neighbor_pos)
                        q.append((neighbor_pos, new_path))
        return None

    def _update_particles(self, time_factor):
        processed_count = 0
        for p in self.particles[:]:
            p.progress += p.speed * time_factor / self.GRID_SIZE
            start_pos = (p.x, p.y)
            
            if p.progress >= 1.0:
                p.progress = 0.0
                p.x, p.y = p.target_pos['x'], p.target_pos['y']
                
                # Process particle at component
                current_comp = self.components.get(p.target_pos['grid_pos'])
                if current_comp:
                    if current_comp.type == "PROCESSOR":
                        if p.type == "RAW": p.type = "PROCESSED"
                        # Sound: Processing sizzle
                    elif current_comp.type == "SINK":
                        # Check if sink accepts this particle type
                        if p.type == "PROCESSED":
                            processed_count += 1
                        self.particles.remove(p)
                        continue
                
                if not p.path:
                    self.particles.remove(p)
                    continue
                p.target_pos = p.path.popleft()

            else:
                current_x = start_pos[0] + (p.target_pos['x'] - start_pos[0]) * p.progress
                current_y = start_pos[1] + (p.target_pos['y'] - start_pos[1]) * p.progress
                p.trail.append((current_x, current_y))
        
        return processed_count


    # --- RENDERING METHODS ---
    def _create_fractal_surface(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.COLOR_BG)
        self.buildable_rects = []

        def draw_carpet(rect, depth):
            if depth == 0:
                self.buildable_rects.append(rect)
                return
            
            w = rect.width / 3
            h = rect.height / 3
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        color = self.COLOR_FRACTAL_2
                        pygame.draw.rect(surface, color, (rect.x + w, rect.y + h, w, h))
                    else:
                        color = self.COLOR_FRACTAL_1
                        sub_rect = pygame.Rect(rect.x + i * w, rect.y + j * h, w, h)
                        pygame.draw.rect(surface, color, sub_rect)
                        draw_carpet(sub_rect, depth - 1)

        initial_rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT)
        draw_carpet(initial_rect, 2) # Depth 2 is a good balance
        return surface

    def _is_buildable(self, grid_pos):
        pixel_x = grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2
        pixel_y = grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        for r in self.buildable_rects:
            if r.collidepoint(pixel_x, pixel_y):
                return True
        return False

    def _render_game(self):
        # Draw components
        for comp in self.components.values():
            color = self.COMPONENT_COLORS[comp.type]
            if not comp.is_powered and comp.type not in ["SOURCE", "SINK"]:
                color = (70, 70, 80)
            
            if comp.type == "PROCESSOR":
                pygame.gfxdraw.filled_circle(self.screen, int(comp.x), int(comp.y), self.GRID_SIZE // 2 - 2, color)
                if comp.is_powered:
                    angle = (pygame.time.get_ticks() / 1000.0) * math.pi
                    radius = (self.GRID_SIZE // 2 - 4) * (0.9 + 0.1 * math.sin(angle))
                    pygame.gfxdraw.aacircle(self.screen, int(comp.x), int(comp.y), int(radius), (255,255,255))
            else:
                rect = pygame.Rect(comp.x - self.GRID_SIZE // 2, comp.y - self.GRID_SIZE // 2, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            color = self.RESOURCE_COLORS[p.type]
            # Trail
            if p.trail:
                for i, pos in enumerate(p.trail):
                    alpha = int(255 * (i / len(p.trail)))
                    trail_color = (*color, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, trail_color)
                # Head
                pygame.gfxdraw.filled_circle(self.screen, int(p.trail[-1][0]), int(p.trail[-1][1]), 4, color)
                pygame.gfxdraw.aacircle(self.screen, int(p.trail[-1][0]), int(p.trail[-1][1]), 4, (255,255,255))

        # Draw flying components
        for fc in self.flying_components:
            size = int(self.GRID_SIZE * (fc.progress * 0.5 + 0.5))
            color = (*self.COMPONENT_COLORS[fc.type], int(255 * (1-fc.progress)))
            rect = pygame.Rect(int(fc.current_pos[0] - size//2), int(fc.current_pos[1] - size//2), size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw aimer and trajectory
        aimer_x = self.aimer_grid_pos[0] * self.GRID_SIZE
        aimer_y = self.aimer_grid_pos[1] * self.GRID_SIZE
        aimer_rect = pygame.Rect(aimer_x, aimer_y, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_AIMER, aimer_rect, 1)
        pygame.draw.aaline(self.screen, self.COLOR_TRAJECTORY, self.launcher_pos, (aimer_x + self.GRID_SIZE//2, aimer_y + self.GRID_SIZE//2))

    def _render_ui(self):
        # Time slow effect
        if self.time_slow_active:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIMESLOW_OVERLAY)
            self.screen.blit(overlay, (0,0))
            self._draw_text("TIME SLOW", self.font_m, (self.WIDTH // 2, 30))

        # Score and Throughput
        score_text = f"Score: {self.score:.1f}"
        throughput_text = f"TPM: {self.max_throughput_achieved:.0f}"
        self._draw_text(score_text, self.font_m, (10, 10), "topleft")
        self._draw_text(throughput_text, self.font_m, (10, 40), "topleft")

        # Selected Component
        selected_text = f"Build: {self.COMPONENT_TYPES[self.selected_component_idx]}"
        self._draw_text(selected_text, self.font_m, (self.WIDTH - 10, self.HEIGHT - 10), "bottomright")

    def _draw_text(self, text, font, pos, anchor="center"):
        text_surf = font.render(text, True, self.COLOR_UI_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_UI_SHADOW)
        text_rect = text_surf.get_rect()
        if anchor == "center": text_rect.center = pos
        elif anchor == "topleft": text_rect.topleft = pos
        elif anchor == "bottomright": text_rect.bottomright = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch back to a visible driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.quit()
    pygame.init()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal Factory")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Max Throughput: {env.max_throughput_achieved:.0f}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()