import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# --- Constants ---
# Game world
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
GRID_WIDTH, GRID_HEIGHT = 20, 15
TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 32, 16
WORLD_ORIGIN_X, WORLD_ORIGIN_Y = SCREEN_WIDTH // 2, 60

# Path definition (grid coordinates)
ENEMY_PATH = [(x, 5) for x in range(GRID_WIDTH)] + [(GRID_WIDTH - 1, y) for y in range(6, GRID_HEIGHT)]


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move cursor. Space to place tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in an isometric 2D environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Class Constants ---
    MAX_STEPS = 3000 # Increased for longer games
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_PATH = (50, 60, 70)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLACEABLE = (60, 80, 60, 100)
    COLOR_BASE = (0, 150, 200)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_CURSOR_INVALID = (255, 0, 0, 150)
    
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_UI_BG = (10, 15, 20, 200)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR = (20, 200, 20)
    
    # Tower definitions
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 100, "range": 120, "damage": 5, "fire_rate": 5, "color": (0, 180, 255)},
        1: {"name": "Cannon", "cost": 250, "range": 180, "damage": 25, "fire_rate": 30, "color": (255, 100, 0)}
    }
    
    # Enemy definitions
    ENEMY_SPECS = {
        0: {"name": "Grunt", "health": 50, "speed": 1.0, "reward": 10, "color": (220, 50, 50)},
        1: {"name": "Tank", "health": 200, "speed": 0.6, "reward": 30, "color": (150, 30, 30)}
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.placeable_tiles = self._get_placeable_tiles()
        
        # This will be initialized in reset
        self.game_state = {}

        self.reset()
        self.validate_implementation()

    def _world_to_screen(self, x, y):
        screen_x = WORLD_ORIGIN_X + (x - y) * TILE_WIDTH_HALF
        screen_y = WORLD_ORIGIN_Y + (x + y) * TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_placeable_tiles(self):
        path_set = set(ENEMY_PATH)
        placeable = set()
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if (x, y) not in path_set:
                    placeable.add((x, y))
        return placeable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "game_over": False,
            "victory": False,
            "reward_this_step": 0.0,
            
            "base_health": 100,
            "resources": 250,
            
            "current_wave": 0,
            "wave_in_progress": False,
            "wave_cooldown": 150, # 5 seconds at 30fps
            "enemies_to_spawn": [],
            "spawn_cooldown": 0,
            
            "enemies": [],
            "towers": [],
            "projectiles": [],
            "particles": [],
            
            "cursor_pos": [GRID_WIDTH // 2, GRID_HEIGHT // 2 - 3],
            "selected_tower": 0,
            
            "last_space_held": False,
            "last_shift_held": False,
        }
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        gs = self.game_state
        gs["reward_this_step"] = 0.0
        gs["game_over"] = (gs["base_health"] <= 0) or (gs["steps"] >= self.MAX_STEPS) or gs["victory"]
        
        if not gs["game_over"]:
            self._handle_input(action)
            self._update_waves()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
        
        self._update_particles()
        
        gs["steps"] += 1
        
        terminated = gs["game_over"]
        if terminated and not gs["victory"] and gs["base_health"] > 0:
             gs["reward_this_step"] -= 50 # Penalty for timeout
        
        reward = gs["reward_this_step"]
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        gs = self.game_state
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: gs["cursor_pos"][1] -= 1
        elif movement == 2: gs["cursor_pos"][1] += 1
        elif movement == 3: gs["cursor_pos"][0] -= 1
        elif movement == 4: gs["cursor_pos"][0] += 1
        gs["cursor_pos"][0] = np.clip(gs["cursor_pos"][0], 0, GRID_WIDTH - 1)
        gs["cursor_pos"][1] = np.clip(gs["cursor_pos"][1], 0, GRID_HEIGHT - 1)
        
        # Cycle tower type
        if shift_held and not gs["last_shift_held"]:
            gs["selected_tower"] = (gs["selected_tower"] + 1) % len(self.TOWER_SPECS)
        gs["last_shift_held"] = shift_held
        
        # Place tower
        if space_held and not gs["last_space_held"]:
            self._try_place_tower()
        gs["last_space_held"] = space_held

    def _try_place_tower(self):
        gs = self.game_state
        cx, cy = gs["cursor_pos"]
        spec = self.TOWER_SPECS[gs["selected_tower"]]
        
        is_placeable = (cx, cy) in self.placeable_tiles
        is_occupied = any(t["pos"] == [cx, cy] for t in gs["towers"])
        has_resources = gs["resources"] >= spec["cost"]
        
        if is_placeable and not is_occupied and has_resources:
            gs["resources"] -= spec["cost"]
            gs["towers"].append({
                "pos": [cx, cy],
                "type": gs["selected_tower"],
                "cooldown": 0,
                "spec": spec
            })
            # Visual effect for placement
            self._create_particles(self._world_to_screen(cx, cy), 20, spec["color"], 15)
            # Sound: build_tower.wav
        else:
            # Sound: action_fail.wav
            pass

    def _update_waves(self):
        gs = self.game_state
        if gs["wave_in_progress"]:
            if not gs["enemies"] and not gs["enemies_to_spawn"]:
                gs["wave_in_progress"] = False
                gs["wave_cooldown"] = 300 # 10s cooldown
                gs["reward_this_step"] += 1.0 * gs["current_wave"]
                gs["score"] += 100 * gs["current_wave"]
                # Sound: wave_complete.wav
                if gs["current_wave"] >= self.MAX_WAVES:
                    gs["victory"] = True
                    gs["game_over"] = True
                    gs["reward_this_step"] += 100
        else:
            gs["wave_cooldown"] -= 1
            if gs["wave_cooldown"] <= 0 and gs["current_wave"] < self.MAX_WAVES:
                gs["current_wave"] += 1
                gs["wave_in_progress"] = True
                self._generate_wave(gs["current_wave"])

    def _generate_wave(self, wave_num):
        gs = self.game_state
        num_grunts = 3 + wave_num * 2
        num_tanks = wave_num // 2
        
        wave_list = [0] * num_grunts + [1] * num_tanks
        random.shuffle(wave_list)
        
        for enemy_type in wave_list:
            spec = self.ENEMY_SPECS[enemy_type]
            health_multiplier = 1 + (wave_num - 1) * 0.05
            speed_multiplier = 1 + (wave_num - 1) * 0.05
            
            gs["enemies_to_spawn"].append({
                "type": enemy_type,
                "max_health": spec["health"] * health_multiplier,
                "speed": spec["speed"] * speed_multiplier
            })
            
    def _update_enemies(self):
        gs = self.game_state
        # Spawn new enemies
        if gs["wave_in_progress"] and gs["enemies_to_spawn"]:
            gs["spawn_cooldown"] -= 1
            if gs["spawn_cooldown"] <= 0:
                enemy_data = gs["enemies_to_spawn"].pop(0)
                spec = self.ENEMY_SPECS[enemy_data["type"]]
                
                gs["enemies"].append({
                    "type": enemy_data["type"],
                    "pos": list(ENEMY_PATH[0]),
                    "pixel_pos": self._world_to_screen(*ENEMY_PATH[0]),
                    "path_index": 0,
                    "health": enemy_data["max_health"],
                    "max_health": enemy_data["max_health"],
                    "speed": enemy_data["speed"],
                    "spec": spec
                })
                gs["spawn_cooldown"] = 30 # 1s between spawns

        # Move existing enemies
        for enemy in reversed(gs["enemies"]):
            if enemy["path_index"] >= len(ENEMY_PATH) - 1:
                gs["base_health"] -= enemy["spec"]["reward"] # Damage base proportional to enemy value
                gs["reward_this_step"] -= 2.0
                gs["enemies"].remove(enemy)
                # Sound: base_damage.wav
                if gs["base_health"] <= 0:
                    gs["base_health"] = 0
                    gs["game_over"] = True
                    gs["reward_this_step"] -= 100
                continue
            
            target_node = ENEMY_PATH[enemy["path_index"] + 1]
            target_pixel_pos = self._world_to_screen(*target_node)
            
            current_pixel_pos = list(enemy["pixel_pos"])
            direction = np.array(target_pixel_pos) - np.array(current_pixel_pos)
            dist = np.linalg.norm(direction)
            
            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                enemy["pos"] = ENEMY_PATH[enemy["path_index"]]
                enemy["pixel_pos"] = self._world_to_screen(*enemy["pos"])
            else:
                direction_norm = direction / dist
                move = direction_norm * enemy["speed"]
                enemy["pixel_pos"] = (current_pixel_pos[0] + move[0], current_pixel_pos[1] + move[1])
                
    def _update_towers(self):
        gs = self.game_state
        for tower in gs["towers"]:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = self._find_target(tower)
                if target:
                    tower["cooldown"] = tower["spec"]["fire_rate"]
                    gs["projectiles"].append({
                        "start_pos": self._world_to_screen(*tower["pos"]),
                        "target_enemy": target,
                        "pos": self._world_to_screen(*tower["pos"]),
                        "damage": tower["spec"]["damage"],
                        "color": tower["spec"]["color"],
                        "speed": 15
                    })
                    # Sound: tower_shoot.wav

    def _find_target(self, tower):
        gs = self.game_state
        tower_pos = np.array(self._world_to_screen(*tower["pos"]))
        in_range_enemies = []
        for enemy in gs["enemies"]:
            enemy_pos = np.array(enemy["pixel_pos"])
            dist = np.linalg.norm(tower_pos - enemy_pos)
            if dist <= tower["spec"]["range"]:
                in_range_enemies.append(enemy)
        
        # Target enemy closest to the base
        if not in_range_enemies:
            return None
        return max(in_range_enemies, key=lambda e: e["path_index"])

    def _update_projectiles(self):
        gs = self.game_state
        for proj in reversed(gs["projectiles"]):
            target_pos = np.array(proj["target_enemy"]["pixel_pos"])
            proj_pos = np.array(proj["pos"])
            
            direction = target_pos - proj_pos
            dist = np.linalg.norm(direction)
            
            if dist < proj["speed"]:
                self._hit_enemy(proj)
                gs["projectiles"].remove(proj)
            else:
                move = (direction / dist) * proj["speed"]
                proj["pos"] = (proj_pos[0] + move[0], proj_pos[1] + move[1])

    def _hit_enemy(self, projectile):
        gs = self.game_state
        enemy = projectile["target_enemy"]
        
        # Create impact particles
        self._create_particles(enemy["pixel_pos"], 10, projectile["color"], 10)
        # Sound: projectile_hit.wav

        if enemy not in gs["enemies"]: return # Target already dead

        enemy["health"] -= projectile["damage"]
        if enemy["health"] <= 0:
            gs["reward_this_step"] += 0.1
            gs["score"] += enemy["spec"]["reward"]
            gs["resources"] += enemy["spec"]["reward"]
            gs["enemies"].remove(enemy)
            # Sound: enemy_die.wav
            self._create_particles(enemy["pixel_pos"], 30, enemy["spec"]["color"], 20, 5)

    def _update_particles(self):
        gs = self.game_state
        for p in reversed(gs["particles"]):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                gs["particles"].remove(p)

    def _create_particles(self, pos, count, color, lifetime, speed=3):
        gs = self.game_state
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, speed)
            vel = [math.cos(angle) * s, math.sin(angle) * s]
            gs["particles"].append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(lifetime // 2, lifetime),
                "color": color,
                "size": random.randint(2, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                screen_pos = self._world_to_screen(x, y)
                tile_points = [
                    self._world_to_screen(x, y),
                    self._world_to_screen(x + 1, y),
                    self._world_to_screen(x + 1, y + 1),
                    self._world_to_screen(x, y + 1)
                ]
                color = self.COLOR_PATH if (x, y) in ENEMY_PATH else self.COLOR_GRID
                pygame.gfxdraw.filled_polygon(self.screen, tile_points, color)
                pygame.gfxdraw.aapolygon(self.screen, tile_points, color)
        
        # Draw placeable area overlay
        placeable_surf = self.screen.copy()
        placeable_surf.set_colorkey((0,0,0))
        for x, y in self.placeable_tiles:
            tile_points = [
                self._world_to_screen(x, y),
                self._world_to_screen(x + 1, y),
                self._world_to_screen(x + 1, y + 1),
                self._world_to_screen(x, y + 1)
            ]
            pygame.gfxdraw.filled_polygon(placeable_surf, tile_points, self.COLOR_PLACEABLE)
        placeable_surf.set_alpha(self.COLOR_PLACEABLE[3])
        self.screen.blit(placeable_surf, (0,0))

        # Draw base
        base_pos = ENEMY_PATH[-1]
        base_points = [
            self._world_to_screen(base_pos[0], base_pos[1]),
            self._world_to_screen(base_pos[0] + 1, base_pos[1]),
            self._world_to_screen(base_pos[0] + 1, base_pos[1] + 1),
            self._world_to_screen(base_pos[0], base_pos[1] + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, tuple(min(255, c+40) for c in self.COLOR_BASE))

        # Collect and sort all dynamic objects for correct Z-ordering
        render_queue = []
        for t in self.game_state["towers"]:
            render_queue.append(("tower", t))
        for e in self.game_state["enemies"]:
            render_queue.append(("enemy", e))
        
        render_queue.sort(key=lambda item: self._world_to_screen(*item[1]["pos"])[1])

        # Draw sorted objects
        for item_type, item in render_queue:
            if item_type == "tower":
                self._draw_tower(item)
            elif item_type == "enemy":
                self._draw_enemy(item)

        # Draw projectiles and particles on top
        for p in self.game_state["projectiles"]:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 3, p["color"])
        for p in self.game_state["particles"]:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, p["pos"], p["size"])
        
        # Draw cursor
        self._draw_cursor()

    def _draw_tower(self, tower):
        pos = self._world_to_screen(*tower["pos"])
        spec = tower["spec"]
        pygame.draw.circle(self.screen, spec["color"], (pos[0], pos[1] - TILE_HEIGHT_HALF), 8)
        pygame.draw.rect(self.screen, (50,50,50), (pos[0]-5, pos[1]-TILE_HEIGHT_HALF, 10, 5))

    def _draw_enemy(self, enemy):
        pos = (int(enemy["pixel_pos"][0]), int(enemy["pixel_pos"][1]))
        spec = enemy["spec"]
        
        # Draw body
        size = 10 if enemy["type"] == 0 else 15
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1] - size // 2, size, spec["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1] - size // 2, size, tuple(min(255, c+50) for c in spec["color"]))

        # Draw health bar
        bar_width = 30
        bar_height = 4
        bar_y = pos[1] - size - 10
        health_ratio = enemy["health"] / enemy["max_health"]
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_width // 2, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_width // 2, bar_y, int(bar_width * health_ratio), bar_height))

    def _draw_cursor(self):
        gs = self.game_state
        cx, cy = gs["cursor_pos"]
        screen_pos = self._world_to_screen(cx, cy)
        
        is_placeable = (cx, cy) in self.placeable_tiles
        is_occupied = any(t["pos"] == [cx, cy] for t in gs["towers"])
        
        color = self.COLOR_CURSOR if (is_placeable and not is_occupied) else self.COLOR_CURSOR_INVALID
        
        cursor_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw placement footprint
        tile_points = [
            self._world_to_screen(cx, cy),
            self._world_to_screen(cx + 1, cy),
            self._world_to_screen(cx + 1, cy + 1),
            self._world_to_screen(cx, cy + 1)
        ]
        pygame.gfxdraw.filled_polygon(cursor_surf, tile_points, color)
        
        # Draw range indicator
        spec = self.TOWER_SPECS[gs["selected_tower"]]
        pygame.gfxdraw.aacircle(cursor_surf, screen_pos[0], screen_pos[1], spec["range"], color)
        
        self.screen.blit(cursor_surf, (0,0))
        
    def _render_ui(self):
        gs = self.game_state
        # UI Background Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40))

        # Helper to draw text with shadow
        def draw_text(text, pos, font, color=self.COLOR_TEXT):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, pos)

        # Base Health
        draw_text(f"Base HP: {int(gs['base_health'])}/100", (10, SCREEN_HEIGHT - 30), self.font_s)
        # Resources
        draw_text(f"Resources: ${gs['resources']}", (160, SCREEN_HEIGHT - 30), self.font_s)
        # Wave
        wave_text = f"Wave: {gs['current_wave']}/{self.MAX_WAVES}" if gs["current_wave"] > 0 else "Wave: 0/10"
        draw_text(wave_text, (310, SCREEN_HEIGHT - 30), self.font_s)
        
        # Selected Tower
        spec = self.TOWER_SPECS[gs["selected_tower"]]
        can_afford = gs['resources'] >= spec['cost']
        cost_color = self.COLOR_TEXT if can_afford else (255, 80, 80)
        draw_text(f"Tower: {spec['name']} (Cost: ${spec['cost']})", (430, SCREEN_HEIGHT - 30), self.font_s, cost_color)

        # Game Over / Victory Message
        if gs["game_over"]:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            message = "VICTORY!" if gs["victory"] else "GAME OVER"
            color = (100, 255, 100) if gs["victory"] else (255, 100, 100)
            draw_text(message, (SCREEN_WIDTH/2 - self.font_l.size(message)[0]/2, 150), self.font_l, color)
            
    def _get_info(self):
        return {
            "score": self.game_state["score"],
            "steps": self.game_state["steps"],
            "base_health": self.game_state["base_health"],
            "resources": self.game_state["resources"],
            "wave": self.game_state["current_wave"],
        }
    
    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    display_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Render to display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()