
# Generated: 2025-08-28T05:05:27.927593
# Source Brief: brief_05452.md
# Brief Index: 5452

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class Guard:
    """Helper class to manage guard state and behavior."""
    def __init__(self, path, speed, vision_range, vision_fov):
        self.path = path
        self.speed = speed
        self.vision_range = vision_range
        self.vision_fov = vision_fov
        self.path_index = 0
        self.pos = np.array(self.path[self.path_index], dtype=float)
        self.direction_vector = np.array([1.0, 0.0])
        self.vision_cone_points = []
        self.update_direction()

    def update_direction(self):
        if len(self.path) > 1:
            target_pos = np.array(self.path[self.path_index])
            direction = target_pos - self.pos
            distance = np.linalg.norm(direction)
            if distance > 1:
                self.direction_vector = direction / distance
            # If at the target, update target
            if distance < self.speed:
                self.path_index = (self.path_index + 1) % len(self.path)

    def move(self):
        self.update_direction()
        self.pos += self.direction_vector * self.speed
        self._update_vision_cone()

    def _update_vision_cone(self):
        angle = math.atan2(self.direction_vector[1], self.direction_vector[0])
        p1 = self.pos
        p2 = self.pos + self.vision_range * np.array([math.cos(angle - self.vision_fov / 2), math.sin(angle - self.vision_fov / 2)])
        p3 = self.pos + self.vision_range * np.array([math.cos(angle + self.vision_fov / 2), math.sin(angle + self.vision_fov / 2)])
        self.vision_cone_points = [p1, p2, p3]

    def can_see(self, player_pos, player_rect, shadows):
        # Basic check: player must not be in shadow
        in_shadow = any(player_rect.colliderect(shadow) for shadow in shadows)
        if in_shadow:
            return False

        # Check if player is within vision range
        if np.linalg.norm(player_pos - self.pos) > self.vision_range:
            return False

        # Check if player is inside the vision cone using barycentric coordinates
        p = player_pos
        p0, p1, p2 = self.vision_cone_points
        area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
        if area == 0: return False
        s = 1 / (2 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1])
        t = 1 / (2 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1])
        return s > 0 and t > 0 and 1 - s - t > 0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Stay in the shadows to avoid detection and reach the white exit."
    )
    game_description = (
        "Evade patrolling guards in a procedurally generated top-down stealth environment to reach the exit."
    )
    auto_advance = False

    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_WALL = (60, 60, 75)
    COLOR_SHADOW = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (255, 255, 255)
    COLOR_EXIT = (255, 255, 255)
    COLOR_GUARD = (255, 190, 0)
    COLOR_GUARD_VISION = (255, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.grid_size = 20
        self.grid_w, self.grid_h = self.width // self.grid_size, self.height // self.grid_size
        
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.player_pos = np.array([0.0, 0.0])
        self.player_radius = 6
        self.player_speed = 4.0
        self.player_rect = pygame.Rect(0, 0, 0, 0)

        self.guards = []
        self.walls = []
        self.shadows = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        
        self.max_time = 60.0
        self.time_per_step = 0.1
        self.max_spotted = 3
        
        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        # 1. Create grid and walls using randomized DFS
        grid = [[True for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        stack = deque([(0, 0)])
        visited = set([(0, 0)])
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and (nx, ny) not in visited:
                    neighbors.append((nx, ny, cx + dx, cy + dy))
            
            if neighbors:
                nx, ny, wx, wy = random.choice(neighbors)
                grid[ny][nx] = False
                grid[wy][wx] = False
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # 2. Convert grid to wall rects
        self.walls = []
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if grid[y][x]:
                    self.walls.append(pygame.Rect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size))

        # 3. Find floor cells
        floor_cells = []
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if not grid[y][x]:
                    floor_cells.append((x, y))
        
        # 4. Place shadows
        self.shadows = []
        num_shadows = int(len(floor_cells) * 0.25)
        for _ in range(num_shadows):
            x, y = random.choice(floor_cells)
            self.shadows.append(pygame.Rect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size))

        # 5. Place player and exit
        start_cell = random.choice(floor_cells)
        self.player_pos = np.array([start_cell[0] * self.grid_size + self.grid_size/2, start_cell[1] * self.grid_size + self.grid_size/2])
        
        # Find furthest point for exit using BFS
        q = deque([(start_cell, 0)])
        visited_bfs = {start_cell}
        furthest_cell, max_dist = start_cell, 0
        
        while q:
            (cx, cy), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                furthest_cell = (cx, cy)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and not grid[ny][nx] and (nx, ny) not in visited_bfs:
                    visited_bfs.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        
        self.exit_rect = pygame.Rect(furthest_cell[0] * self.grid_size, furthest_cell[1] * self.grid_size, self.grid_size, self.grid_size)

        # 6. Create guards
        self.guards = []
        num_guards = 3
        for _ in range(num_guards):
            path = []
            for _ in range(random.randint(2, 4)):
                cell = random.choice(floor_cells)
                path.append((cell[0] * self.grid_size + self.grid_size/2, cell[1] * self.grid_size + self.grid_size/2))
            
            self.guards.append(Guard(path, speed=1.5, vision_range=100, vision_fov=math.radians(60)))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.max_time
        self.spotted_count = 0
        self.last_spot_step = -100 # Cooldown to prevent multi-spotting in one moment

        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        self._update_player(movement)
        self._update_guards()

        reward, spotted_this_step = self._calculate_reward()
        self.score += reward

        if spotted_this_step and (self.steps - self.last_spot_step > 10):
            # sfx: alert sound
            self.spotted_count += 1
            self.last_spot_step = self.steps

        self.time_remaining -= self.time_per_step
        self.steps += 1
        
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        move_vector = np.array([0.0, 0.0])
        if movement == 1: move_vector[1] -= 1 # Up
        elif movement == 2: move_vector[1] += 1 # Down
        elif movement == 3: move_vector[0] -= 1 # Left
        elif movement == 4: move_vector[0] += 1 # Right

        if np.linalg.norm(move_vector) > 0:
            # sfx: player footstep
            move_vector = move_vector / np.linalg.norm(move_vector)
            
        new_pos = self.player_pos + move_vector * self.player_speed
        
        # Wall collision
        temp_rect = pygame.Rect(0, 0, self.player_radius * 2, self.player_radius * 2)
        
        # X-axis collision
        temp_rect.center = (new_pos[0], self.player_pos[1])
        if temp_rect.collidelist(self.walls) == -1:
            self.player_pos[0] = new_pos[0]
            
        # Y-axis collision
        temp_rect.center = (self.player_pos[0], new_pos[1])
        if temp_rect.collidelist(self.walls) == -1:
            self.player_pos[1] = new_pos[1]

        # Clamp to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.width - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.height - self.player_radius)
        
        self.player_rect.center = tuple(self.player_pos)
        self.player_rect.size = (self.player_radius*2, self.player_radius*2)

    def _update_guards(self):
        for guard in self.guards:
            guard.move()
    
    def _calculate_reward(self):
        reward = 0
        spotted_this_step = False

        # Check for spotting
        for guard in self.guards:
            if guard.can_see(self.player_pos, self.player_rect, self.shadows):
                reward -= 1.0
                spotted_this_step = True
                break # Only get spotted by one guard at a time

        # Reward for being in shadow, penalty for being in light
        in_shadow = any(self.player_rect.colliderect(shadow) for shadow in self.shadows)
        if in_shadow:
            reward += 0.1
        else:
            reward -= 0.02
        
        # Goal rewards are handled in _check_termination
        return reward, spotted_this_step

    def _check_termination(self):
        if self.player_rect.colliderect(self.exit_rect):
            self.score += 100
            self.game_over = True
            self.game_won = True
            # sfx: win jingle
            return True
        if self.spotted_count >= self.max_spotted:
            self.score -= 100
            self.game_over = True
            # sfx: lose sound
            return True
        if self.time_remaining <= 0:
            self.score -= 100
            self.game_over = True
            # sfx: lose sound
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw shadows
        for shadow in self.shadows:
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow)
            
        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw guards and vision cones
        for guard in self.guards:
            # Vision cone
            if guard.vision_cone_points:
                points = [(int(p[0]), int(p[1])) for p in guard.vision_cone_points]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GUARD_VISION)
                pygame.gfxdraw.filled_polygon(self.screen, points, (*self.COLOR_GUARD_VISION, 40))
            # Guard body
            pos = (int(guard.pos[0]), int(guard.pos[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.player_radius-1, self.COLOR_GUARD)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.player_radius-1, self.COLOR_GUARD)

        # Draw player
        player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius + 3, (*self.COLOR_PLAYER_GLOW, 60))
        # Player body
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time remaining
        time_text = f"Time: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Spotted count
        spotted_text = f"Spotted: {self.spotted_count}/{self.max_spotted}"
        spotted_surf = self.font_ui.render(spotted_text, True, self.COLOR_UI_TEXT)
        spotted_rect = spotted_surf.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(spotted_surf, spotted_rect)

        # Game over text
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text = "MISSION COMPLETE" if self.game_won else "MISSION FAILED"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            text_surf = self.font_game_over.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "spotted_count": self.spotted_count,
            "player_pos": self.player_pos.tolist(),
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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a dummy video driver to run pygame headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play with keyboard ---
    # This is a simple manual control loop for testing
    # Requires pygame to be installed with display support
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Stealth Game")
        
        obs, info = env.reset()
        terminated = False
        clock = pygame.time.Clock()
        
        while not terminated:
            action = [0, 0, 0] # no-op by default
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert observation back to a surface for display
            # The observation is (H, W, C), but pygame wants (W, H) surface
            # Transpose is needed: (W, H, C) -> (H, W, C) -> (W, H, C)
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Limit to 30 FPS for playability
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()
                terminated = False

    except pygame.error as e:
        print("Pygame display could not be initialized. Running headless test.")
        print(f"Error: {e}")
        # Run a simple headless loop
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print("Headless episode finished.")
                obs, info = env.reset()
    
    env.close()