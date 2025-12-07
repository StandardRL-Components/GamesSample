import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:57:16.125643
# Source Brief: brief_01291.md
# Brief Index: 1291
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Plant and match crops to earn resources for building defenses against waves of incoming bugs."
    user_guide = "Use arrow keys (↑↓←→) to move the cursor. Press space to plant/build and shift to cycle tools."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.TILE_SIZE = 40
        self.MAX_STEPS = 5000
        self.WIN_WAVE = 20
        self.HOME_BASE_COL = 0

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_HOME = (50, 30, 30)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_WAVE_BAR = (0, 100, 200)

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Define crops and tools
        self.CROP_DATA = {
            "carrot": {"unlock_wave": 1, "color": (255, 140, 0), "grow_time": 200, "res_yield": (1, 2)},
            "corn": {"unlock_wave": 5, "color": (255, 225, 50), "grow_time": 300, "res_yield": (2, 4)},
            "pumpkin": {"unlock_wave": 10, "color": (200, 80, 0), "grow_time": 450, "res_yield": (3, 6)},
        }
        self.DEFENSE_DATA = {
            "turret": {"cost": 10, "color": (0, 150, 255), "range": 3, "fire_rate": 60, "health": 100}
        }
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.grid = [[{"type": "empty"} for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.bugs = []
        self.defenses = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.resources = {"seeds": 5, "defense_points": 0}
        
        self.wave_num = 0
        self.wave_timer = 300 # Time until the first wave
        self.wave_cooldown = 600

        self.unlocked_tools = []
        self._update_unlocked_tools()
        self.selected_tool_idx = 0

        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Handle input (edge-triggered for space/shift)
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        action_reward = self._handle_input(movement, space_press, shift_press)
        reward += action_reward

        # Update game logic
        self._update_crops()
        update_reward = self._update_defenses()
        reward += update_reward
        self._update_projectiles()
        bug_reward = self._update_bugs()
        reward += bug_reward
        self._update_particles()
        wave_reward = self._update_waves()
        reward += wave_reward
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        # Cycle tool
        if shift_press and len(self.unlocked_tools) > 0:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.unlocked_tools)
            # SFX: UI_Cycle.wav

        # Use tool
        if space_press and len(self.unlocked_tools) > 0:
            tool_name = self.unlocked_tools[self.selected_tool_idx]
            cx, cy = self.cursor_pos
            if tool_name.startswith("plant_"):
                crop_type = tool_name.split("_")[1]
                return self._plant_crop(cx, cy, crop_type)
            elif tool_name == "build_turret":
                return self._build_defense(cx, cy)
        return 0

    def _plant_crop(self, x, y, crop_type):
        if self.grid[y][x]["type"] == "empty" and self.resources["seeds"] > 0:
            self.resources["seeds"] -= 1
            self.grid[y][x] = {"type": "crop", "crop_type": crop_type, "growth": 0}
            # SFX: Plant.wav
            return self._check_matches(x, y, crop_type)
        return 0

    def _check_matches(self, x, y, crop_type):
        mature_neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                tile = self.grid[ny][nx]
                if tile["type"] == "crop" and tile["crop_type"] == crop_type and tile["growth"] >= self.CROP_DATA[crop_type]["grow_time"]:
                    mature_neighbors.append((nx, ny))
        
        if len(mature_neighbors) >= 2:
            all_matched_tiles = [(x, y)] + mature_neighbors
            total_reward = 0
            for mx, my in all_matched_tiles:
                self.grid[my][mx] = {"type": "empty"}
                seed_gain, dp_gain = self.CROP_DATA[crop_type]["res_yield"]
                self.resources["seeds"] += seed_gain
                self.resources["defense_points"] += dp_gain
                total_reward += 0.1
                self._create_particles(mx, my, self.CROP_DATA[crop_type]["color"], 20)
            # SFX: Match.wav
            return total_reward
        return 0

    def _build_defense(self, x, y):
        cost = self.DEFENSE_DATA["turret"]["cost"]
        if self.grid[y][x]["type"] == "empty" and self.resources["defense_points"] >= cost:
            self.resources["defense_points"] -= cost
            turret_info = self.DEFENSE_DATA["turret"]
            self.defenses.append({
                "x": x, "y": y, "health": turret_info["health"], "cooldown": 0,
                "range_sq": turret_info["range"] ** 2, "fire_rate": turret_info["fire_rate"]
            })
            self.grid[y][x] = {"type": "defense"}
            # SFX: Build.wav
            return 0.5 # Small reward for building
        return 0

    def _update_crops(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                tile = self.grid[y][x]
                if tile["type"] == "crop":
                    tile["growth"] = min(self.CROP_DATA[tile["crop_type"]]["grow_time"], tile["growth"] + 1)

    def _update_defenses(self):
        for defense in self.defenses:
            if defense["cooldown"] > 0:
                defense["cooldown"] -= 1
                continue
            
            target = None
            min_dist_sq = defense["range_sq"]
            for bug in self.bugs:
                dist_sq = (bug["x"] - (defense["x"] + 0.5))**2 + (bug["y"] - (defense["y"] + 0.5))**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = bug
            
            if target:
                self.projectiles.append({
                    "x": (defense["x"] + 0.5) * self.TILE_SIZE,
                    "y": (defense["y"] + 0.5) * self.TILE_SIZE,
                    "target": target, "speed": 5, "color": (150, 255, 150)
                })
                defense["cooldown"] = defense["fire_rate"]
                # SFX: Turret_Fire.wav
        return 0

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.bugs:
                self.projectiles.remove(proj)
                continue
            
            target_x = (proj["target"]["x"] + 0.5) * self.TILE_SIZE
            target_y = (proj["target"]["y"] + 0.5) * self.TILE_SIZE
            angle = math.atan2(target_y - proj["y"], target_x - proj["x"])
            proj["x"] += proj["speed"] * math.cos(angle)
            proj["y"] += proj["speed"] * math.sin(angle)
            
            if math.hypot(proj["x"] - target_x, proj["y"] - target_y) < self.TILE_SIZE / 2:
                proj["target"]["health"] -= 25
                self.projectiles.remove(proj)
                if proj["target"]["health"] <= 0:
                    if proj["target"] in self.bugs: # Check if not already removed
                        self.bugs.remove(proj["target"])
                    self._create_particles(proj["target"]["x"], proj["target"]["y"], (200, 50, 50), 15)
                    # SFX: Bug_Die.wav

    def _update_bugs(self):
        reward = 0
        for bug in self.bugs[:]:
            # Simple pathfinding: move left
            bug["x"] -= bug["speed"]
            if bug["x"] * self.TILE_SIZE < self.HOME_BASE_COL * self.TILE_SIZE + self.TILE_SIZE:
                self.bugs.remove(bug)
                self.game_over = True
                reward -= 0.1
                # SFX: Base_Damage.wav
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1 # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _update_waves(self):
        reward = 0
        if self.wave_num >= self.WIN_WAVE:
            return 0

        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.wave_num += 1
            self.wave_timer = self.wave_cooldown
            reward += 1.0 # Wave survived reward
            
            if self.wave_num == self.WIN_WAVE:
                self.win = True
                self.game_over = True

            # Spawn bugs
            num_bugs = 3 + self.wave_num
            bug_speed = 0.01 + 0.005 * (self.wave_num // 2)
            bug_health = 100 + 5 * (self.wave_num // 2)
            for _ in range(num_bugs):
                self.bugs.append({
                    "x": self.GRID_COLS - 1,
                    "y": self.np_random.integers(0, self.GRID_ROWS),
                    "health": bug_health, "speed": bug_speed,
                    "anim_offset": self.np_random.random() * 10
                })
            
            # Check for unlocks
            prev_unlocked_count = len(self.unlocked_tools)
            self._update_unlocked_tools()
            if len(self.unlocked_tools) > prev_unlocked_count:
                reward += 5.0 # Unlock reward
        return reward

    def _update_unlocked_tools(self):
        self.unlocked_tools = []
        for crop, data in self.CROP_DATA.items():
            if self.wave_num >= data["unlock_wave"]:
                self.unlocked_tools.append(f"plant_{crop}")
        self.unlocked_tools.append("build_turret")


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw home base area
        home_rect = pygame.Rect(self.HOME_BASE_COL * self.TILE_SIZE, 0, self.TILE_SIZE, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_HOME, home_rect)

        # Draw grid
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Draw crops, defenses
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                tile = self.grid[y][x]
                if tile["type"] == "crop":
                    self._render_crop(x, y, tile)
        
        for defense in self.defenses:
            self._render_defense(defense)

        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj["color"], (int(proj["x"]), int(proj["y"])), 4)
            
        for bug in self.bugs:
            self._render_bug(bug)

        for p in self.particles:
            size = max(1, int(p["size"] * (p["lifespan"] / p["max_lifespan"])))
            pygame.draw.circle(self.screen, p["color"], (int(p["x"]), int(p["y"])), size)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.TILE_SIZE, cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        # Glow effect
        glow_size = int(self.TILE_SIZE * 1.4)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_CURSOR, 50), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (cursor_rect.centerx - glow_size//2, cursor_rect.centery - glow_size//2))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)


    def _render_crop(self, x, y, crop_tile):
        data = self.CROP_DATA[crop_tile["crop_type"]]
        growth_ratio = crop_tile["growth"] / data["grow_time"]
        
        px, py = (x + 0.5) * self.TILE_SIZE, (y + 0.5) * self.TILE_SIZE
        
        if growth_ratio < 1.0:
            size = int(2 + 10 * growth_ratio)
            pygame.draw.circle(self.screen, (30, 150, 30), (px, py + (10 - size//2)), size)
        else: # Mature crop
            size = int(self.TILE_SIZE * 0.7)
            bob = math.sin(self.steps / 20.0 + x) * 2
            pygame.draw.rect(self.screen, data["color"], (px - size//2, py - size//2 + bob, size, size), border_radius=4)

    def _render_defense(self, defense):
        data = self.DEFENSE_DATA["turret"]
        px, py = (defense["x"] + 0.5) * self.TILE_SIZE, (defense["y"] + 0.5) * self.TILE_SIZE
        base_size = int(self.TILE_SIZE * 0.8)
        top_size = int(self.TILE_SIZE * 0.5)
        pygame.draw.rect(self.screen, data["color"], (px - base_size//2, py - base_size//2, base_size, base_size), border_radius=3)
        pygame.draw.rect(self.screen, (200, 200, 220), (px - top_size//2, py - top_size//2, top_size, top_size), border_radius=3)
        if defense["cooldown"] > defense["fire_rate"] - 5: # Muzzle flash
            pygame.draw.circle(self.screen, (255,255,100), (int(px), int(py)), 10)

    def _render_bug(self, bug):
        px, py = (bug["x"] + 0.5) * self.TILE_SIZE, (bug["y"] + 0.5) * self.TILE_SIZE
        size = int(self.TILE_SIZE * 0.6)
        leg_bob = math.sin((self.steps + bug["anim_offset"]) / 5.0) * 4
        
        # Legs
        pygame.draw.line(self.screen, (100,0,0), (px - leg_bob, py-size//2), (px + leg_bob, py+size//2), 3)
        pygame.draw.line(self.screen, (100,0,0), (px + leg_bob, py-size//2), (px - leg_bob, py+size//2), 3)
        
        # Body
        pygame.draw.ellipse(self.screen, (200, 50, 50), (px - size//2, py - size//3, size, size*0.66))
        # Eyes
        pygame.draw.circle(self.screen, (255, 255, 255), (int(px + size*0.2), int(py)), 2)


    def _render_ui(self):
        # Top resource bar
        pygame.draw.rect(self.screen, (20, 30, 40, 200), (0, 0, self.WIDTH, 30))
        
        seed_text = self.font_small.render(f"Seeds: {self.resources['seeds']}", True, self.COLOR_TEXT)
        self.screen.blit(seed_text, (10, 5))
        
        dp_text = self.font_small.render(f"DP: {self.resources['defense_points']}", True, self.COLOR_TEXT)
        self.screen.blit(dp_text, (150, 5))

        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 150, 5))
        
        # Bottom wave bar
        pygame.draw.rect(self.screen, (20, 30, 40, 200), (0, self.HEIGHT - 30, self.WIDTH, 30))
        wave_text = self.font_small.render(f"Wave {self.wave_num}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, self.HEIGHT - 25))
        
        if self.wave_timer > 0 and self.wave_num < self.WIN_WAVE:
            progress = 1.0 - (self.wave_timer / self.wave_cooldown)
            bar_width = (self.WIDTH - 120) * progress
            pygame.draw.rect(self.screen, self.COLOR_GRID, (110, self.HEIGHT - 22, self.WIDTH - 120, 14))
            pygame.draw.rect(self.screen, self.COLOR_WAVE_BAR, (110, self.HEIGHT - 22, bar_width, 14))
        
        # Selected tool
        if len(self.unlocked_tools) > 0:
            tool_name = self.unlocked_tools[self.selected_tool_idx]
            tool_text = self.font_small.render(f"Tool: {tool_name.replace('_', ' ').title()}", True, self.COLOR_TEXT)
            self.screen.blit(tool_text, (self.WIDTH // 2 - tool_text.get_width()//2, 5))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_CURSOR if self.win else (200, 50, 50))
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        overlay.blit(text, text_rect)
        
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "resources": self.resources,
        }

    def _check_termination(self):
        return self.game_over

    def _create_particles(self, grid_x, grid_y, color, count):
        px, py = (grid_x + 0.5) * self.TILE_SIZE, (grid_y + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                "x": px, "y": py,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "lifespan": self.np_random.integers(20, 40),
                "max_lifespan": 40,
                "color": color,
                "size": self.np_random.integers(3, 7)
            })

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a visible display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Key mapping for manual play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a display screen for manual play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crop Defender")
    
    clock = pygame.time.Clock()
    
    while not done:
        # Action defaults
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get key presses
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()