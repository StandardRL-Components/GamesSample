import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from collections import deque
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Escape a haunted opera house by collecting instruments to craft sonic tools, "
        "all while evading ghostly guardians and falling debris."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Collect instruments to craft tools. "
        "Press Shift to cycle through your crafted tools and Space to use the selected one."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TILE_SIZE = 40
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (20, 15, 30)
    COLOR_WALL_MAIN = (50, 40, 60)
    COLOR_WALL_ACCENT = (70, 60, 80)
    COLOR_FLOOR_MAIN = (35, 30, 45)
    COLOR_FLOOR_ACCENT = (45, 40, 55)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_EXIT = (0, 255, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_ALERT = (255, 150, 0)
    COLOR_DEBRIS = (150, 120, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (10, 5, 20, 180)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (200, 50, 50)

    # Map Layout
    MAP_LAYOUT = [
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
        "W P  I        W     D      W     W     I        W",
        "W WW W WWWWWW W WWWWWWWWWW W WWW W WWWWWWWWWWWW W",
        "W  W   W   W  W W          W   W W W     I    W W",
        "W WWWWWW W W WW W WWWWWWWWWW W W W W WWWWWWWW W W",
        "W W    T W W  W   W          W W   W        W W W",
        "W W WWWW W WWWWWW W WWWWWWWW W WWWWWWWWWWWW W W W",
        "W W    W W      W W        W W     W        W W W",
        "W WWWW W WWWWWW W WWWWWWWW W WWWWWWWWWWWWWW W W W",
        "W      W   I  W W        W W     W   D    W   W W",
        "W WWWWWWWWWW WW WWWWWWWW W W WWW W WWWWWWWWWW W W",
        "W W        W  W        W W W W W W          W W W",
        "W W WWWWWW W WWWWWWWWWW W W W W W WWWWWWWWWW W W W",
        "W   W    W W D        W W W W   W   I      W W W",
        "W WWW WW W WWWWWWWWWW W W W WWWWWWWWWWWWWW W W W",
        "W D   W  W          W   W W              W W W W",
        "W WWWWW WWWWWWWWWWWWWWWW W WWWWWWWWWWWWWW W W W W",
        "W T                  I W W                W E W",
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    ]
    MAP_WIDTH = len(MAP_LAYOUT[0])
    MAP_HEIGHT = len(MAP_LAYOUT)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)

        # Game state variables
        self.player_grid_pos = np.array([0, 0])
        self.player_render_pos = np.array([0.0, 0.0])
        self.player_health = 100
        self.max_health = 100
        self.inventory = {} # item_name: count
        self.tools = [] # tool_name
        self.selected_tool_idx = -1
        
        self.exit_pos = np.array([0, 0])
        self.walls = []
        self.instruments = []
        self.enemies = []
        self.debris = []
        self.effects = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.prev_space_held = False
        self.prev_shift_held = False

        self.debris_spawn_chance = 0.05
        self.enemy_speed_multiplier = 1.0

        self.camera_offset = np.array([0.0, 0.0])
        
        # Crafting recipes: {tool_name: {ingredient: count}}
        self.recipes = {
            "Trumpet Stun": {"Trumpet": 1},
            "Sonic Lure": {"Violin": 1, "Flute": 1},
            "Rhythmic Shield": {"Drum": 2},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.max_health
        self.inventory = {}
        self.tools = []
        self.selected_tool_idx = -1

        self.walls.clear()
        self.instruments.clear()
        self.enemies.clear()
        self.debris.clear()
        self.effects.clear()
        
        instrument_spawns = []
        drum_spawns = []
        trumpet_spawns = []

        for y, row in enumerate(self.MAP_LAYOUT):
            for x, char in enumerate(row):
                pos = np.array([x, y])
                if char == 'W':
                    self.walls.append(pos)
                elif char == 'P':
                    self.player_grid_pos = pos
                elif char == 'E':
                    self.exit_pos = pos
                elif char == 'I':
                    instrument_spawns.append(pos)
                elif char == 'D':
                    drum_spawns.append(pos)
                elif char == 'T':
                    trumpet_spawns.append(pos)
        
        # Spawn Instruments
        available_instruments = ["Violin", "Flute"]
        instrument_choices = list(available_instruments)
        self.np_random.shuffle(instrument_choices)
        
        for pos in instrument_spawns:
            if not instrument_choices:
                instrument_choices = list(available_instruments)
                self.np_random.shuffle(instrument_choices)
            self.instruments.append({"pos": pos, "type": instrument_choices.pop()})
        for pos in drum_spawns:
            self.instruments.append({"pos": pos, "type": "Drum"})
        for pos in trumpet_spawns:
            self.instruments.append({"pos": pos, "type": "Trumpet"})

        # Spawn Enemies (simple patrol)
        enemy_starts = [np.array([15, 3]), np.array([25, 13]), np.array([8, 16])]
        for start_pos in enemy_starts:
            self.enemies.append({
                "grid_pos": start_pos.copy(), "render_pos": start_pos.astype(float) * self.TILE_SIZE,
                "path": [start_pos.copy(), start_pos.copy() + np.array([8, 0])], "path_idx": 0,
                "state": "patrol", "stun_timer": 0, "move_cooldown": 0,
                "target_pos": start_pos.copy()
            })

        self.player_render_pos = self.player_grid_pos.astype(float) * self.TILE_SIZE
        self.debris_spawn_chance = 0.05
        self.enemy_speed_multiplier = 1.0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # 1. Player Movement & Rewards
        old_dist_to_exit = np.linalg.norm(self.player_grid_pos - self.exit_pos)
        
        target_pos = self.player_grid_pos.copy()
        if movement == 1: target_pos[1] -= 1  # Up
        elif movement == 2: target_pos[1] += 1 # Down
        elif movement == 3: target_pos[0] -= 1 # Left
        elif movement == 4: target_pos[0] += 1 # Right
        
        if movement != 0 and not any(np.array_equal(target_pos, w) for w in self.walls):
            self.player_grid_pos = target_pos

        new_dist_to_exit = np.linalg.norm(self.player_grid_pos - self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.1

        # 2. Tool Actions
        if shift_pressed and self.tools:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.tools)

        if space_pressed and self.selected_tool_idx != -1 and self.tools:
            tool_name = self.tools.pop(self.selected_tool_idx)
            self._use_tool(tool_name)
            self.selected_tool_idx = -1 if not self.tools else 0

        # 3. Game World Updates
        self._update_enemies()
        self._update_debris()
        
        # 4. Collisions & Interactions
        # Instrument collection
        for inst in self.instruments[:]:
            if np.array_equal(self.player_grid_pos, inst["pos"]):
                self.inventory[inst["type"]] = self.inventory.get(inst["type"], 0) + 1
                self.instruments.remove(inst)
                reward += 1.0
                self.score += 10
                self._add_effect(inst["pos"] * self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_EXIT, "+1 Inst", 30)
        
        # Crafting
        crafted_tool = self._check_crafting()
        if crafted_tool:
            self.tools.append(crafted_tool)
            if self.selected_tool_idx == -1: self.selected_tool_idx = 0
            reward += 2.0
            self.score += 20
            self._add_effect(self.player_grid_pos * self.TILE_SIZE, self.COLOR_PLAYER, "Crafted!", 45)

        # Debris collision
        player_rect = pygame.Rect(self.player_grid_pos * self.TILE_SIZE, (self.TILE_SIZE, self.TILE_SIZE))
        for d in self.debris[:]:
            debris_rect = pygame.Rect(d["pos"], (d["size"], d["size"]))
            if player_rect.colliderect(debris_rect):
                shielded = False
                for effect in self.effects:
                    if effect["type"] == "shield":
                        effect["life"] = 0 # Consume shield
                        shielded = True
                        break
                if not shielded:
                    self.player_health -= 10
                    reward -= 0.5
                self.debris.remove(d)
                break
        
        # Enemy collision
        for enemy in self.enemies:
            if np.array_equal(self.player_grid_pos, enemy["grid_pos"]):
                self.player_health -= 25
                reward -= 1.0

        # 5. Difficulty Progression
        self.steps += 1
        if self.steps % 10 == 0: self.debris_spawn_chance += 0.005
        if self.steps % 100 == 0: self.enemy_speed_multiplier = max(0.5, self.enemy_speed_multiplier - 0.05)

        # 6. Termination Conditions
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward = -100
            terminated = True
            self.game_over = True
        elif np.array_equal(self.player_grid_pos, self.exit_pos):
            reward = 100
            self.score += 1000
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["stun_timer"] > 0:
                enemy["stun_timer"] -= 1
                continue
            
            enemy["move_cooldown"] -= 1
            if enemy["move_cooldown"] > 0:
                continue

            dist_to_player = np.linalg.norm(self.player_grid_pos - enemy["grid_pos"])
            if dist_to_player < 6:
                enemy["state"] = "alert"
                # Simple chase logic
                move_dir = self.player_grid_pos - enemy["grid_pos"]
                next_pos = enemy["grid_pos"].copy()
                # Move on the axis with the largest difference
                if abs(move_dir[0]) > abs(move_dir[1]):
                    next_pos[0] += np.sign(move_dir[0])
                elif abs(move_dir[1]) > abs(move_dir[0]):
                    next_pos[1] += np.sign(move_dir[1])
                
                if not any(np.array_equal(next_pos, w) for w in self.walls):
                    enemy["grid_pos"] = next_pos
                enemy["move_cooldown"] = int(20 * self.enemy_speed_multiplier)

            else:
                enemy["state"] = "patrol"
                target_path_pos = enemy["path"][enemy["path_idx"]]
                if np.array_equal(enemy["grid_pos"], target_path_pos):
                    enemy["path_idx"] = 1 - enemy["path_idx"] # Flip between 0 and 1
                    target_path_pos = enemy["path"][enemy["path_idx"]]

                move_dir = target_path_pos - enemy["grid_pos"]
                if np.any(move_dir):
                    move_dir = np.sign(move_dir)
                    next_pos = enemy["grid_pos"] + move_dir
                    if not any(np.array_equal(next_pos, w) for w in self.walls):
                        enemy["grid_pos"] = next_pos
                enemy["move_cooldown"] = int(30 * self.enemy_speed_multiplier)

    def _update_debris(self):
        if self.np_random.random() < self.debris_spawn_chance:
            x = self.np_random.integers(0, self.MAP_WIDTH * self.TILE_SIZE)
            size = self.np_random.integers(5, 15)
            self.debris.append({
                "pos": np.array([float(x), -float(size)]),
                "speed": self.np_random.uniform(1.0, 3.0),
                "size": size,
            })
        
        for d in self.debris[:]:
            d["pos"][1] += d["speed"]
            if d["pos"][1] > self.HEIGHT:
                self.debris.remove(d)

    def _check_crafting(self):
        for tool_name, ingredients in self.recipes.items():
            can_craft = True
            for item, required_count in ingredients.items():
                if self.inventory.get(item, 0) < required_count:
                    can_craft = False
                    break
            if can_craft:
                # Consume ingredients
                for item, required_count in ingredients.items():
                    self.inventory[item] -= required_count
                return tool_name
        return None

    def _use_tool(self, tool_name):
        if tool_name == "Trumpet Stun":
            self._add_effect(self.player_grid_pos * self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_PLAYER, "STUN!", 60, "stun_wave")
            for enemy in self.enemies:
                if np.linalg.norm(self.player_grid_pos - enemy["grid_pos"]) < 5:
                    enemy["stun_timer"] = 150 # 5 seconds at 30fps
        elif tool_name == "Sonic Lure":
            self._add_effect(self.player_grid_pos * self.TILE_SIZE, self.COLOR_EXIT, "LURE", 300, "lure")
        elif tool_name == "Rhythmic Shield":
            self._add_effect(self.player_grid_pos * self.TILE_SIZE, self.COLOR_PLAYER, "SHIELD", 300, "shield")

    def _add_effect(self, pos, color, text, life, effect_type="text"):
        self.effects.append({
            "pos": np.array(pos, dtype=float), "color": color, "text": text,
            "life": life, "max_life": life, "type": effect_type
        })

    def _get_observation(self):
        # --- Interpolate positions for smooth rendering ---
        lerp_factor = 0.4
        target_player_pos = self.player_grid_pos.astype(float) * self.TILE_SIZE
        self.player_render_pos = self.player_render_pos * (1 - lerp_factor) + target_player_pos * lerp_factor
        
        for enemy in self.enemies:
            target_enemy_pos = enemy["grid_pos"].astype(float) * self.TILE_SIZE
            enemy["render_pos"] = enemy["render_pos"] * (1 - lerp_factor) + target_enemy_pos * lerp_factor

        # --- Camera ---
        self.camera_offset = self.player_render_pos - np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.camera_offset[0] = np.clip(self.camera_offset[0], 0, self.MAP_WIDTH * self.TILE_SIZE - self.WIDTH)
        self.camera_offset[1] = np.clip(self.camera_offset[1], 0, self.MAP_HEIGHT * self.TILE_SIZE - self.HEIGHT)
        
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        cam_x, cam_y = self.camera_offset

        # Visible tile range
        start_x = int(cam_x / self.TILE_SIZE)
        end_x = int((cam_x + self.WIDTH) / self.TILE_SIZE) + 1
        start_y = int(cam_y / self.TILE_SIZE)
        end_y = int((cam_y + self.HEIGHT) / self.TILE_SIZE) + 1

        # Render Floor & Walls
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if 0 <= y < self.MAP_HEIGHT and 0 <= x < self.MAP_WIDTH:
                    screen_pos = (x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y)
                    rect = pygame.Rect(screen_pos, (self.TILE_SIZE, self.TILE_SIZE))
                    
                    if self.MAP_LAYOUT[y][x] == 'W':
                        pygame.draw.rect(self.screen, self.COLOR_WALL_MAIN, rect)
                        pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, rect, 1)
                    else:
                        color = self.COLOR_FLOOR_MAIN if (x+y)%2 == 0 else self.COLOR_FLOOR_ACCENT
                        pygame.draw.rect(self.screen, color, rect)

        # Render Exit
        exit_screen_pos = self.exit_pos * self.TILE_SIZE - self.camera_offset
        exit_rect = pygame.Rect(exit_screen_pos, (self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, (255,255,255), exit_rect, 2)

        # Render Instruments
        for inst in self.instruments:
            pos = inst["pos"] * self.TILE_SIZE - self.camera_offset
            center = (int(pos[0] + self.TILE_SIZE/2), int(pos[1] + self.TILE_SIZE/2))
            color = {"Violin": (255,255,0), "Flute": (200,200,255), "Drum": (200,100,50), "Trumpet": (255, 215, 0)}[inst["type"]]
            pygame.draw.circle(self.screen, color, center, int(self.TILE_SIZE/3))
            pygame.draw.circle(self.screen, (255,255,255), center, int(self.TILE_SIZE/3), 1)

        # Render Enemies
        for enemy in self.enemies:
            pos = enemy["render_pos"] - self.camera_offset
            center = (int(pos[0] + self.TILE_SIZE/2), int(pos[1] + self.TILE_SIZE/2))
            radius = int(self.TILE_SIZE/2.5)
            
            if enemy["stun_timer"] > 0:
                pygame.draw.circle(self.screen, (100, 100, 255), center, radius)
            else:
                pygame.draw.circle(self.screen, self.COLOR_ENEMY, center, radius)

            if enemy["state"] == "alert" and enemy["stun_timer"] == 0:
                pygame.draw.circle(self.screen, self.COLOR_ENEMY_ALERT, center, radius + 3, 2)

        # Render Player
        player_center = (int(self.player_render_pos[0] - cam_x + self.TILE_SIZE/2), int(self.player_render_pos[1] - cam_y + self.TILE_SIZE/2))
        for i in range(10, 0, -2): # Glow effect
            alpha = 100 - i * 10
            glow_color = (*self.COLOR_PLAYER_GLOW, alpha)
            s = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (i, i), i)
            self.screen.blit(s, (player_center[0] - i, player_center[1] - i))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_center, int(self.TILE_SIZE/2.2))

        # Render Effects
        for effect in self.effects[:]:
            effect["life"] -= 1
            if effect["life"] <= 0:
                self.effects.remove(effect)
                continue
            
            alpha = int(255 * (effect["life"] / effect["max_life"]))
            if effect["type"] == "text":
                text_surf = self.font_small.render(effect["text"], True, effect["color"])
                text_surf.set_alpha(alpha)
                pos = effect["pos"] - self.camera_offset
                self.screen.blit(text_surf, (pos[0], pos[1] - (20 * (1 - effect["life"]/effect["max_life"]))))
            elif effect["type"] == "stun_wave":
                radius = int(self.TILE_SIZE * 5 * (1 - effect["life"]/effect["max_life"]))
                pos = effect["pos"] - self.camera_offset
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, (*effect["color"], alpha))
            elif effect["type"] == "shield":
                s_alpha = 100 + 100 * math.sin(self.steps * 0.3)
                pygame.draw.circle(self.screen, (*self.COLOR_PLAYER, int(s_alpha)), player_center, int(self.TILE_SIZE/2 + 3), 2)

        # Render Debris
        for d in self.debris:
            pos = d["pos"] - self.camera_offset
            pygame.draw.rect(self.screen, self.COLOR_DEBRIS, (pos[0], pos[1], d["size"], d["size"]))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.max_health)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"Health: {self.player_health}/{self.max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score and Steps
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 65))

        # Inventory / Tools
        inv_surface = pygame.Surface((180, 120), pygame.SRCALPHA)
        inv_surface.fill(self.COLOR_UI_BG)
        
        y_offset = 5
        # Draw collected instruments
        for item, count in self.inventory.items():
            if count > 0:
                text = f"{item}: {count}"
                text_surf = self.font_small.render(text, True, self.COLOR_TEXT)
                inv_surface.blit(text_surf, (5, y_offset))
                y_offset += 18
        
        # Draw crafted tools
        if self.tools:
            y_offset += 5
            pygame.draw.line(inv_surface, self.COLOR_TEXT, (5, y_offset), (175, y_offset), 1)
            y_offset += 5
            for i, tool in enumerate(self.tools):
                color = self.COLOR_PLAYER if i == self.selected_tool_idx else self.COLOR_TEXT
                text_surf = self.font_small.render(f"> {tool}", True, color)
                inv_surface.blit(text_surf, (5, y_offset))
                y_offset += 18

        self.screen.blit(inv_surface, (self.WIDTH - 190, self.HEIGHT - 130))

        # Mini-map
        map_surf_size = 100
        map_surf = pygame.Surface((map_surf_size, map_surf_size), pygame.SRCALPHA)
        map_surf.fill(self.COLOR_UI_BG)
        scale = map_surf_size / (self.MAP_WIDTH * self.TILE_SIZE)
        
        # Draw map elements scaled down
        for y, row in enumerate(self.MAP_LAYOUT):
            for x, char in enumerate(row):
                if char == 'W':
                    pygame.draw.rect(map_surf, self.COLOR_WALL_ACCENT, (x * self.TILE_SIZE * scale, y * self.TILE_SIZE * scale, 2, 2))

        # Player on minimap
        px, py = self.player_grid_pos
        pygame.draw.rect(map_surf, self.COLOR_PLAYER, (px * self.TILE_SIZE * scale, py * self.TILE_SIZE * scale, 3, 3))
        # Exit on minimap
        ex, ey = self.exit_pos
        pygame.draw.rect(map_surf, self.COLOR_EXIT, (ex * self.TILE_SIZE * scale, ey * self.TILE_SIZE * scale, 3, 3))

        self.screen.blit(map_surf, (self.WIDTH - map_surf_size - 10, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This part is for human testing and is not part of the gym environment.
    # It demonstrates how to control the environment.
    # For this to work, you might need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Opera House Escape")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    terminated = False
    truncated = False
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False
                total_reward = 0

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated or truncated:
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            if terminated:
                text = "YOU ESCAPED!" if info["health"] > 0 else "YOU PERISHED"
                color = GameEnv.COLOR_EXIT if info["health"] > 0 else GameEnv.COLOR_ENEMY
            else: # Truncated
                text = "TIME'S UP!"
                color = GameEnv.COLOR_ENEMY_ALERT

            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 - 20))
            screen.blit(text_surf, text_rect)

            font_small = pygame.font.SysFont("Consolas", 20)
            score_text = f"Final Score: {int(info['score'])}"
            score_surf = font_small.render(score_text, True, GameEnv.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 + 20))
            screen.blit(score_surf, score_rect)

            reset_text = font_small.render("Press 'R' to restart", True, GameEnv.COLOR_TEXT)
            reset_rect = reset_text.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 + 50))
            screen.blit(reset_text, reset_rect)


        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS for consistent game speed

    env.close()