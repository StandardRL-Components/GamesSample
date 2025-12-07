import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:11:28.230389
# Source Brief: brief_02373.md
# Brief Index: 2373
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifetime"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Combine crystals of the same type to increase their strength. Terraform the grid to create fusion bonuses and unlock more powerful crystal types."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to select/deselect a crystal, move it to an empty cell, or fuse it with an adjacent one. Press 'shift' to terraform the land for a fusion bonus."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (40, 50, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TERRAFORM = (0, 255, 100)
    
    CRYSTAL_SPECS = [
        {"color": (50, 150, 255), "sides": 3},  # Type 0: Blue Triangle
        {"color": (255, 255, 50), "sides": 4},  # Type 1: Yellow Square
        {"color": (255, 50, 255), "sides": 5},  # Type 2: Magenta Pentagon
        {"color": (50, 255, 255), "sides": 6},  # Type 3: Cyan Hexagon
        {"color": (255, 120, 0), "sides": 8},  # Type 4: Orange Octagon
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_info = pygame.font.SysFont("monospace", 14)

        # --- Grid & Layout ---
        self.grid_area_width = 380
        self.grid_area_height = 380
        self.cell_size = self.grid_area_width / self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_width) / 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_height) / 2

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_crystal_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.unlocked_types = []
        self.next_unlock_score = 0
        self.particles = []
        self.last_reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_crystal_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.last_reward_info = {}
        
        self.unlocked_types = [0]
        self.next_unlock_score = 500

        self.grid = [
            [{"crystal": None, "terraformed": False} for _ in range(self.GRID_SIZE)]
            for _ in range(self.GRID_SIZE)
        ]

        for _ in range(3):
            self._spawn_crystal(crystal_type=0, strength=10)
        
        self._calculate_total_strength()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        self.steps += 1
        reward = 0.0
        self.last_reward_info = {}

        # 1. Handle Input & Game Logic
        self._handle_movement(movement)
        
        if shift_press:
            reward += self._attempt_terraform()

        if space_press:
            reward += self._handle_selection()

        # 2. Update Game State
        prev_score_tier = math.floor(self.score / 100)
        self._calculate_total_strength()
        new_score_tier = math.floor(self.score / 100)
        
        if new_score_tier > prev_score_tier:
            score_reward = (new_score_tier - prev_score_tier) * 10
            reward += score_reward
            self.last_reward_info["Strength Milestone"] = score_reward

        # Check for unlocking new crystal types
        if self.score >= self.next_unlock_score and len(self.unlocked_types) < len(self.CRYSTAL_SPECS):
            new_type = len(self.unlocked_types)
            self.unlocked_types.append(new_type)
            self.next_unlock_score += 500
            unlock_reward = 5.0
            reward += unlock_reward
            self.last_reward_info["New Crystal"] = unlock_reward

        # Spawn new crystals periodically
        if self.steps % 10 == 0:
            spawned_type = self.np_random.choice(self.unlocked_types)
            self._spawn_crystal(crystal_type=spawned_type, strength=10 * (spawned_type + 1))
        
        self._update_particles()

        # 3. Check Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

    def _attempt_terraform(self):
        x, y = self.cursor_pos
        cell = self.grid[y][x]
        
        terraform_cost = 25
        if not cell["terraformed"] and self.score >= terraform_cost:
            cell["terraformed"] = True
            self.score -= terraform_cost # Cost to terraform
            # Sound: Terraform_Activate.wav
            return -0.5 # Small negative reward for spending
        return 0

    def _handle_selection(self):
        reward = 0
        cursor_x, cursor_y = self.cursor_pos
        
        if self.selected_crystal_pos is None:
            # --- Try to select a crystal ---
            if self.grid[cursor_y][cursor_x]["crystal"] is not None:
                self.selected_crystal_pos = (cursor_x, cursor_y)
                # Sound: Select_Crystal.wav
        else:
            # --- Crystal already selected, try to perform an action ---
            start_x, start_y = self.selected_crystal_pos
            
            # Deselect if clicking the same crystal
            if (start_x, start_y) == (cursor_x, cursor_y):
                self.selected_crystal_pos = None
                return 0

            # Check for fusion
            is_adjacent = abs(start_x - cursor_x) + abs(start_y - cursor_y) == 1
            start_crystal = self.grid[start_y][start_x]["crystal"]
            target_crystal = self.grid[cursor_y][cursor_x]["crystal"]

            if is_adjacent and target_crystal and start_crystal["type"] == target_crystal["type"]:
                # --- Perform Fusion ---
                # Sound: Fusion_Success.wav
                bonus = 1.2 if self.grid[cursor_y][cursor_x]["terraformed"] else 1.0
                new_strength = int((start_crystal["strength"] + target_crystal["strength"]) * bonus)
                
                target_crystal["strength"] = new_strength
                self.grid[start_y][start_x]["crystal"] = None
                
                fusion_reward = 1.0
                reward += fusion_reward
                self.last_reward_info["Fusion"] = fusion_reward
                self._add_fusion_particles((cursor_x, cursor_y))
            
            elif target_crystal is None:
                # --- Perform Move ---
                # Sound: Move_Crystal.wav
                self.grid[cursor_y][cursor_x]["crystal"] = start_crystal
                self.grid[start_y][start_x]["crystal"] = None
                if self.grid[cursor_y][cursor_x]["terraformed"]:
                    move_reward = 0.1
                    reward += move_reward
                    self.last_reward_info["Terraform Move"] = move_reward
            else:
                # --- Invalid Action ---
                # Sound: Action_Fail.wav
                pass

            self.selected_crystal_pos = None # Deselect after any action
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        
        empty_cells = 0
        for row in self.grid:
            for cell in row:
                if cell["crystal"] is None:
                    empty_cells += 1
        
        if empty_cells == 0:
            # Check for any possible fusions
            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    crystal = self.grid[y][x]["crystal"]
                    if not crystal: continue
                    # Check neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                            neighbor = self.grid[ny][nx]["crystal"]
                            if neighbor and neighbor["type"] == crystal["type"]:
                                return False # A fusion is possible
            return True # Board is full and no fusions possible

        return False

    def _calculate_total_strength(self):
        total_strength = 0
        for row in self.grid:
            for cell in row:
                if cell["crystal"]:
                    total_strength += cell["crystal"]["strength"]
        self.score = total_strength

    def _spawn_crystal(self, crystal_type, strength):
        empty_cells = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x]["crystal"] is None:
                    empty_cells.append((x, y))
        
        if empty_cells:
            idx = self.np_random.integers(len(empty_cells))
            x, y = empty_cells[idx]
            self.grid[y][x]["crystal"] = {"type": crystal_type, "strength": strength}

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
            "unlocked_types": len(self.unlocked_types),
            "cursor_pos": list(self.cursor_pos),
        }

    def _grid_to_pixel(self, x, y):
        px = self.grid_offset_x + (x + 0.5) * self.cell_size
        py = self.grid_offset_y + (y + 0.5) * self.cell_size
        return int(px), int(py)

    def _render_game(self):
        # Draw grid and terraformed tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                if self.grid[y][x]["terraformed"]:
                    s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    color = self.COLOR_TERRAFORM
                    pygame.draw.rect(s, (*color, 60), s.get_rect())
                    pygame.draw.rect(s, (*color, 180), s.get_rect(), 2)
                    self.screen.blit(s, rect.topleft)

                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw crystals
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell = self.grid[y][x]
                if cell["crystal"]:
                    self._draw_crystal(x, y, cell["crystal"])

        # Draw selection highlight
        if self.selected_crystal_pos:
            x, y = self.selected_crystal_pos
            px, py = self._grid_to_pixel(x, y)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(self.cell_size * 0.5 * (0.9 + pulse * 0.1))
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, (255, 255, 255, 200))
            pygame.gfxdraw.aacircle(self.screen, px, py, radius-1, (255, 255, 255, 200))

        # Draw cursor
        self._draw_cursor()
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), p.color)

    def _draw_crystal(self, x, y, crystal):
        px, py = self._grid_to_pixel(x, y)
        spec = self.CRYSTAL_SPECS[crystal["type"]]
        color = spec["color"]
        sides = spec["sides"]
        
        # Size and glow scale with log of strength to keep sizes reasonable
        base_radius = self.cell_size * 0.1
        log_strength = math.log(max(1, crystal["strength"]))
        radius = int(base_radius + log_strength * 2)
        radius = min(radius, self.cell_size * 0.45) # Cap size

        # Draw glow
        glow_radius = int(radius * (1.5 + log_strength * 0.1))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*color, 50)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main shape
        points = []
        angle_offset = math.pi / 2 if sides % 2 != 0 else math.pi / sides
        for i in range(sides):
            angle = i * (2 * math.pi / sides) - angle_offset
            pt_x = px + radius * math.cos(angle)
            pt_y = py + radius * math.sin(angle)
            points.append((int(pt_x), int(pt_y)))

        if len(points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor(self):
        px, py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        size = int(self.cell_size * 0.5)
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        color = (255, 255, 200, int(150 + pulse * 105))
        line_len = int(size * 0.2)

        # Draw 4 brackets [ ]
        tl = (px - size, py - size)
        tr = (px + size, py - size)
        bl = (px - size, py + size)
        br = (px + size, py + size)
        
        pygame.draw.line(self.screen, color, tl, (tl[0] + line_len, tl[1]), 2)
        pygame.draw.line(self.screen, color, tl, (tl[0], tl[1] + line_len), 2)
        
        pygame.draw.line(self.screen, color, tr, (tr[0] - line_len, tr[1]), 2)
        pygame.draw.line(self.screen, color, tr, (tr[0], tr[1] + line_len), 2)

        pygame.draw.line(self.screen, color, bl, (bl[0] + line_len, bl[1]), 2)
        pygame.draw.line(self.screen, color, bl, (bl[0], bl[1] - line_len), 2)
        
        pygame.draw.line(self.screen, color, br, (br[0] - line_len, br[1]), 2)
        pygame.draw.line(self.screen, color, br, (br[0], br[1] - line_len), 2)

    def _add_fusion_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos[0], grid_pos[1])
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 5)
            color = random.choice([(255, 80, 80), (255, 180, 50), (255, 255, 255)])
            lifetime = self.np_random.integers(20, 40)
            self.particles.append(Particle([px, py], vel, radius, color, lifetime))

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            new_pos = (p.pos[0] + p.vel[0], p.pos[1] + p.vel[1])
            new_lifetime = p.lifetime - 1
            new_radius = p.radius * (new_lifetime / p.lifetime) if p.lifetime > 0 else 0
            if new_lifetime > 0:
                new_particles.append(p._replace(pos=new_pos, lifetime=new_lifetime, radius=new_radius))
        self.particles = new_particles

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"STRENGTH: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Unlocked types display
        type_y = 50
        for i, spec in enumerate(self.CRYSTAL_SPECS):
            is_unlocked = i in self.unlocked_types
            color = spec["color"] if is_unlocked else (80, 80, 80)
            
            # Draw a small sample crystal
            points = []
            sides = spec["sides"]
            radius = 10
            px, py = 35, type_y + 10
            angle_offset = math.pi / 2 if sides % 2 != 0 else math.pi / sides
            for j in range(sides):
                angle = j * (2 * math.pi / sides) - angle_offset
                points.append((px + radius * math.cos(angle), py + radius * math.sin(angle)))
            
            if len(points) > 1:
                pygame.draw.polygon(self.screen, color, points)

            text = "UNLOCKED" if is_unlocked else f"({(i+1)*500} STR)"
            text_surf = self.font_info.render(text, True, color)
            self.screen.blit(text_surf, (60, type_y))
            type_y += 30

        # Last reward info
        reward_y = self.SCREEN_HEIGHT - 20
        for key, value in self.last_reward_info.items():
            reward_text = self.font_info.render(f"{key}: +{value:.1f}", True, (200, 255, 200))
            text_rect = reward_text.get_rect(bottomright=(self.SCREEN_WIDTH - 15, reward_y))
            self.screen.blit(reward_text, text_rect)
            reward_y -= 20


    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'cocoa' depending on your OS
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Fusion")

    # Game loop
    running = True
    while running:
        # --- Pygame event handling ---
        action = np.array([0, 0, 0]) # [movement, space, shift]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # Convert the observation (H, W, C) back to a surface (W, H)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surface.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            running = False # End after one game
            pygame.time.wait(2000)

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()